import csv
import hashlib
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch as pt

from .encoding import encode_features, encode_structure, extract_topology
from .structure import clean_structure, res3to1
from .structure_io import read_pdb


FIELD_SEP = ";"
COMPACT_STRUCTURE_SCHEMA_VERSION = "m4_compact_structure_npz_v3"
SUPPORTED_COMPACT_STRUCTURE_SCHEMA_VERSIONS = {"m4_compact_structure_npz_v2", COMPACT_STRUCTURE_SCHEMA_VERSION}
AA_TO_IDX = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
PAD_IDX = 21


def read_manifest(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_field(text, cast=str):
    if text is None or text == "":
        return []
    return [cast(x) for x in str(text).split(FIELD_SEP)]


def residue_uid(resid, icode=""):
    icode = str(icode or "").strip()
    return f"{int(resid)}{icode}" if icode else str(int(resid))


def structure_residue_ids(structure):
    ids = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode in zip(structure["chain_name"].tolist(), structure["resid"].tolist(), icodes):
        chain = str(chain).split(":")[0]
        key = (chain, int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        ids.append(residue_uid(resid, icode))
    return ids


def structure_residue_chain_ids(structure):
    chain_to_idx = {}
    ids = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode in zip(structure["chain_name"].tolist(), structure["resid"].tolist(), icodes):
        chain = str(chain).split(":")[0]
        key = (chain, int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        if chain not in chain_to_idx:
            chain_to_idx[chain] = len(chain_to_idx)
        ids.append(chain_to_idx[chain])
    return pt.tensor(ids, dtype=pt.long)


@lru_cache(maxsize=8192)
def residue_chain_ids_for_pdb(path):
    try:
        structure = clean_structure(read_pdb(str(path)))
        return structure_residue_chain_ids(structure).cpu()
    except Exception:
        return None


def raw_residue_ids_and_sequence(path):
    structure = read_pdb(str(path))
    ids = []
    seq = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for i, resid in enumerate(structure["resid"]):
        if str(structure["het_flag"][i]) != "A":
            continue
        chain = str(structure["chain_name"][i]).split(":")[0]
        icode = str(icodes[i] or "").strip()
        key = (chain, int(resid), icode)
        if key in seen:
            continue
        seen.add(key)
        ids.append(residue_uid(resid, icode))
        seq.append(res3to1.get(str(structure["resname"][i]).strip().upper(), "X"))
    return ids, seq


def align_polymer_structure_to_residue_ids(structure, residue_ids):
    """Keep polymer atoms for exactly the ordered residue IDs in a manifest row."""
    desired = [str(value).strip() for value in residue_ids if str(value).strip()]
    if not desired:
        raise ValueError("Strict residue alignment requires non-empty manifest residue IDs.")
    if len(desired) != len(set(desired)):
        raise ValueError("Strict residue alignment does not support duplicate manifest residue IDs.")

    icodes = structure.get("icode", [""] * len(structure["resid"]))
    atom_ids = np.asarray([
        residue_uid(resid, icode)
        for resid, icode in zip(structure["resid"], icodes)
    ])
    polymer_mask = structure["het_flag"] == "A"

    raw_ids = []
    raw_sequence = []
    seen = set()
    for index in np.flatnonzero(polymer_mask):
        residue_id = str(atom_ids[index])
        if residue_id in seen:
            continue
        seen.add(residue_id)
        if residue_id in desired:
            raw_ids.append(residue_id)
            raw_sequence.append(
                res3to1.get(str(structure["resname"][index]).strip().upper(), "X")
            )

    if raw_ids != desired:
        missing = [value for value in desired if value not in set(raw_ids)]
        raise ValueError(
            "Manifest residue IDs are not an ordered subset of polymer coordinates: "
            f"manifest={len(desired)} matched={len(raw_ids)} missing={missing[:8]}"
        )

    desired_set = set(desired)
    selected = polymer_mask & np.isin(atom_ids, list(desired_set))
    aligned = {key: value[selected] for key, value in structure.items()}
    if not len(aligned["resid"]):
        raise ValueError("Strict residue alignment removed every structure atom.")
    return aligned, raw_ids, raw_sequence


def sequence_to_tensor(seq):
    ids = [AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"]) for aa in str(seq).strip()]
    if not ids:
        ids = [PAD_IDX]
    return pt.tensor(ids, dtype=pt.long)


def sequence_mask(seq_tensor):
    return (seq_tensor != PAD_IDX).to(pt.float)


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_esm_sequence(seq, max_length=1022):
    seq = "".join((seq or "").strip().upper().split())
    seq = "".join(ch if ch in AA20 else "X" for ch in seq)
    if not seq:
        seq = "X"
    if max_length and len(seq) > max_length:
        seq = seq[:max_length]
    return seq


def esm_sequence_key(sequence, model_name="esm2_t33_650M_UR50D", repr_layer=33, max_length=1022):
    seq = clean_esm_sequence(sequence, max_length=max_length)
    payload = f"{model_name}|layer={repr_layer}|max={max_length}|seq={seq}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def esm_cache_path_for_sequence(
    cache_dir,
    sequence,
    model_name="esm2_t33_650M_UR50D",
    repr_layer=33,
    max_length=1022,
):
    key = esm_sequence_key(sequence, model_name=model_name, repr_layer=repr_layer, max_length=max_length)
    return Path(cache_dir) / key[:2] / f"{key}.pt"


def esm_cache_path_for_key(cache_dir, key):
    key = str(key).strip()
    return Path(cache_dir) / key[:2] / f"{key}.pt"


def structure_sequence_to_tensor(structure):
    seq = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode, resname in zip(
        structure["chain_name"].tolist(),
        structure["resid"].tolist(),
        icodes,
        structure["resname"].tolist(),
    ):
        key = (str(chain).split(":")[0], int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        seq.append(res3to1.get(str(resname).upper(), "X"))
    return sequence_to_tensor("".join(seq))


def structure_sequence_list(structure):
    seq = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode, resname in zip(
        structure["chain_name"].tolist(),
        structure["resid"].tolist(),
        icodes,
        structure["resname"].tolist(),
    ):
        key = (str(chain).split(":")[0], int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        seq.append(res3to1.get(str(resname).strip().upper(), "X"))
    return seq


def cache_key_for_row(row):
    raw = FIELD_SEP.join(
        [
            row.get("apex_case_id", ""),
            row.get("receptor_pdb_path", ""),
            row.get("partner_sequence", ""),
            row.get("input_residue_ids", ""),
            row.get("labels", ""),
            row.get("label_mask", ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in row.get("apex_case_id", "apex"))[:120]
    return f"{stem}_{digest}.pt"


def cache_path_for_row(cache_dir, row):
    return Path(cache_dir) / cache_key_for_row(row)


def compact_structure_identity_for_row(row):
    identity = str(row.get("receptor_pdb_relpath", "") or "").strip()
    if not identity:
        identity = Path(str(row.get("receptor_pdb_path", "") or "")).name
    identity = identity.replace("\\", "/")
    if not identity:
        raise ValueError("Cannot derive compact structure identity from manifest row.")
    return identity


def compact_structure_cache_key_for_row(row):
    identity = compact_structure_identity_for_row(row)
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def compact_structure_cache_path_for_row(cache_dir, row):
    key = compact_structure_cache_key_for_row(row)
    return Path(cache_dir) / key[:2] / key[2:4] / f"{key}.npz"


def _small_unsigned_numpy(tensor):
    array = tensor.detach().cpu().numpy()
    maximum = int(array.max()) if array.size else 0
    if maximum < 0:
        return array.astype(np.int32, copy=False)
    if maximum <= np.iinfo(np.uint8).max:
        return array.astype(np.uint8, copy=False)
    if maximum <= np.iinfo(np.uint16).max:
        return array.astype(np.uint16, copy=False)
    return array.astype(np.uint32, copy=False)


def save_compact_structure_npz(
    path,
    item,
    row,
    residue_graph_cache=None,
    s1_surface_cache=None,
):
    """Save one structure-only, lossless base payload without labels or partner data."""
    path = Path(path)
    assignment = item["M"].detach().cpu().float()
    if assignment.ndim != 2 or assignment.shape[1] <= 0:
        raise ValueError("Compact structure requires a non-empty atom-to-residue matrix.")
    atom_mass = assignment.sum(dim=1)
    if not pt.equal(atom_mass, pt.ones_like(atom_mass)):
        raise ValueError("Compact structure requires one-hot atom-to-residue assignments.")
    atom_to_residue = assignment.argmax(dim=1).long()
    residue_ids = str(row.get("input_residue_ids", "") or "")
    arrays = {
        "schema_version": np.asarray(COMPACT_STRUCTURE_SCHEMA_VERSION),
        "structure_identity": np.asarray(compact_structure_identity_for_row(row)),
        "input_residue_ids_sha1": np.asarray(
            hashlib.sha1(residue_ids.encode("utf-8")).hexdigest() if residue_ids else ""
        ),
        "num_residues": np.asarray(int(assignment.shape[1]), dtype=np.int32),
        "max_num_nn": np.asarray(int(item["ids_topk"].shape[1]), dtype=np.int32),
        "X": item["X"].detach().cpu().numpy(),
        "q": item["q"].detach().cpu().numpy(),
        "atom_to_residue_index": _small_unsigned_numpy(atom_to_residue),
        "ids_topk": _small_unsigned_numpy(item["ids_topk"].long()),
        "receptor_seq_indices": _small_unsigned_numpy(item["receptor_seq_indices"].long()),
        "residue_chain_ids": _small_unsigned_numpy(item["residue_chain_ids"].long()),
    }
    if residue_graph_cache is not None:
        arrays["has_residue_graph"] = np.asarray(1, dtype=np.uint8)
        arrays["graph_ks"] = np.asarray(residue_graph_cache["ks"], dtype=np.uint16)
        arrays["graph_centers"] = residue_graph_cache["centers"].detach().cpu().numpy()
        for k in residue_graph_cache["ks"]:
            arrays[f"graph_knn_idx_k{k}"] = _small_unsigned_numpy(
                residue_graph_cache[f"knn_idx_k{k}"].long()
            )
            arrays[f"graph_knn_dist_k{k}"] = (
                residue_graph_cache[f"knn_dist_k{k}"].detach().cpu().numpy()
            )
    if s1_surface_cache is not None:
        arrays["has_s1_surface"] = np.asarray(1, dtype=np.uint8)
        for name in (
            "surface_pos",
            "surface_normal",
            "surface_geom",
            "surface_graph_dist",
            "surface_lappe",
            "nearest_residue_dist",
            "surface_mask",
        ):
            arrays[name] = s1_surface_cache[name].detach().cpu().numpy()
        for name in ("surface_graph_idx", "nearest_residue_idx"):
            arrays[name] = _small_unsigned_numpy(
                s1_surface_cache[name].long()
            )
        surface_graph_ks = tuple(int(k) for k in s1_surface_cache.get("surface_graph_ks", ()))
        if surface_graph_ks:
            arrays["surface_graph_ks"] = np.asarray(surface_graph_ks, dtype=np.uint16)
            for graph_k in surface_graph_ks:
                idx_name = f"surface_graph_idx_k{graph_k}"
                dist_name = f"surface_graph_dist_k{graph_k}"
                if idx_name in s1_surface_cache and dist_name in s1_surface_cache:
                    arrays[idx_name] = _small_unsigned_numpy(s1_surface_cache[idx_name].long())
                    arrays[dist_name] = s1_surface_cache[dist_name].detach().cpu().numpy()
        arrays["surface_num_surface_points"] = np.asarray(
            int(s1_surface_cache["num_surface_points"]), dtype=np.int32
        )
        arrays["surface_num_residues"] = np.asarray(
            int(s1_surface_cache["num_residues"]), dtype=np.int32
        )
        arrays["surface_cache_kind"] = np.asarray(str(s1_surface_cache["cache_kind"]))
        arrays["surface_probe_radius"] = np.asarray(
            float(s1_surface_cache["probe_radius"]), dtype=np.float64
        )
        arrays["surface_sphere_samples"] = np.asarray(
            int(s1_surface_cache["sphere_samples"]), dtype=np.int32
        )
        arrays["surface_accessible_candidates"] = np.asarray(
            int(s1_surface_cache["accessible_candidates"]), dtype=np.int64
        )
        for name in ("surface_point_density", "surface_area"):
            if name in s1_surface_cache:
                arrays[name] = np.asarray(float(s1_surface_cache[name]), dtype=np.float64)
        if "surface_point_max" in s1_surface_cache:
            arrays["surface_point_max"] = np.asarray(
                int(s1_surface_cache["surface_point_max"]), dtype=np.int32
            )
        for name in ("source_atom_idx", "surface_element_idx"):
            if name in s1_surface_cache:
                arrays[name] = _small_unsigned_numpy(s1_surface_cache[name].long())
        for name in ("surface_chemistry", "surface_curvature", "residue_sasa"):
            if name in s1_surface_cache:
                arrays[name] = s1_surface_cache[name].detach().cpu().numpy()
        for name in ("surface_graph_mode", "surface_point_mode", "semantic_schema"):
            if name in s1_surface_cache:
                arrays[name] = np.asarray(str(s1_surface_cache[name]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_compact_structure_npz(path, row, max_num_nn=64, strict_residue_alignment=True):
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        schema = str(payload["schema_version"].item())
        if schema not in SUPPORTED_COMPACT_STRUCTURE_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported compact structure schema: {schema}")
        expected_identity = compact_structure_identity_for_row(row)
        stored_identity = str(payload["structure_identity"].item())
        if stored_identity != expected_identity:
            raise ValueError(
                f"Compact structure identity mismatch: stored={stored_identity} expected={expected_identity}"
            )
        num_residues = int(payload["num_residues"].item())
        stored_max_num_nn = int(payload["max_num_nn"].item())
        required_max_num_nn = min(int(max_num_nn), int(payload["X"].shape[0]))
        if stored_max_num_nn < required_max_num_nn:
            raise ValueError(
                f"Compact structure topology width {stored_max_num_nn} is smaller than "
                f"the required width {required_max_num_nn}."
            )
        manifest_residue_ids = str(row.get("input_residue_ids", "") or "")
        stored_residue_hash = str(payload["input_residue_ids_sha1"].item())
        if strict_residue_alignment and manifest_residue_ids:
            expected_hash = hashlib.sha1(manifest_residue_ids.encode("utf-8")).hexdigest()
            if stored_residue_hash != expected_hash:
                raise ValueError("Compact structure residue-order hash mismatch.")
        atom_to_residue = pt.from_numpy(payload["atom_to_residue_index"].astype(np.int64)).long()
        if atom_to_residue.numel() and int(atom_to_residue.max().item()) >= num_residues:
            raise ValueError("Compact structure atom-to-residue index is out of range.")
        assignment = pt.nn.functional.one_hot(
            atom_to_residue,
            num_classes=num_residues,
        ).float()
        item = {
            "X": pt.from_numpy(payload["X"].copy()),
            "ids_topk": pt.from_numpy(
                payload["ids_topk"][:, : int(max_num_nn)].astype(np.int64)
            ).long(),
            "q": pt.from_numpy(payload["q"].copy()),
            "M": assignment,
            "receptor_seq_indices": pt.from_numpy(
                payload["receptor_seq_indices"].astype(np.int64)
            ).long(),
            "residue_chain_ids": pt.from_numpy(
                payload["residue_chain_ids"].astype(np.int64)
            ).long(),
        }
        if "has_residue_graph" in payload and int(payload["has_residue_graph"].item()):
            ks = [int(value) for value in payload["graph_ks"].tolist()]
            graph = {
                "centers": pt.from_numpy(payload["graph_centers"].copy()),
                "num_residues": num_residues,
                "ks": ks,
            }
            for k in ks:
                graph[f"knn_idx_k{k}"] = pt.from_numpy(
                    payload[f"graph_knn_idx_k{k}"].astype(np.int64)
                ).long()
                graph[f"knn_dist_k{k}"] = pt.from_numpy(
                    payload[f"graph_knn_dist_k{k}"].copy()
                )
            item["residue_graph_cache"] = graph
        if "has_s1_surface" in payload and int(payload["has_s1_surface"].item()):
            surface = {}
            for name in (
                "surface_pos",
                "surface_normal",
                "surface_geom",
                "surface_graph_dist",
                "surface_lappe",
                "nearest_residue_dist",
                "surface_mask",
            ):
                surface[name] = pt.from_numpy(payload[name].copy())
            for name in ("surface_graph_idx", "nearest_residue_idx"):
                surface[name] = pt.from_numpy(
                    payload[name].astype(np.int64)
                ).long()
            if "surface_graph_ks" in payload:
                surface_graph_ks = tuple(int(k) for k in payload["surface_graph_ks"].tolist())
                surface["surface_graph_ks"] = surface_graph_ks
                for graph_k in surface_graph_ks:
                    idx_name = f"surface_graph_idx_k{graph_k}"
                    dist_name = f"surface_graph_dist_k{graph_k}"
                    surface[idx_name] = pt.from_numpy(payload[idx_name].astype(np.int64)).long()
                    surface[dist_name] = pt.from_numpy(payload[dist_name].copy())
            surface.update(
                {
                    "num_surface_points": int(payload["surface_num_surface_points"].item()),
                    "num_residues": int(payload["surface_num_residues"].item()),
                    "cache_kind": str(payload["surface_cache_kind"].item()),
                    "probe_radius": float(payload["surface_probe_radius"].item()),
                    "sphere_samples": int(payload["surface_sphere_samples"].item()),
                    "accessible_candidates": int(
                        payload["surface_accessible_candidates"].item()
                    ),
                }
            )
            for name in ("source_atom_idx", "surface_element_idx"):
                if name in payload:
                    surface[name] = pt.from_numpy(payload[name].astype(np.int64)).long()
            for name in ("surface_chemistry", "surface_curvature", "residue_sasa"):
                if name in payload:
                    surface[name] = pt.from_numpy(payload[name].copy())
            for name in ("surface_graph_mode", "surface_point_mode", "semantic_schema"):
                if name in payload:
                    surface[name] = str(payload[name].item())
            for name in ("surface_point_density", "surface_area"):
                if name in payload:
                    surface[name] = float(payload[name].item())
            if "surface_point_max" in payload:
                surface["surface_point_max"] = int(payload["surface_point_max"].item())
            item["s1_surface_cache"] = surface

    labels = pt.tensor(split_field(row.get("labels", ""), int), dtype=pt.float)
    mask = pt.tensor(split_field(row.get("label_mask", ""), int), dtype=pt.float)
    if strict_residue_alignment and (labels.numel() != num_residues or mask.numel() != num_residues):
        raise ValueError(
            f"Compact structure residue count mismatch for apex_case_id={row.get('apex_case_id', '')}: "
            f"labels={labels.numel()} mask={mask.numel()} encoded={num_residues}"
        )
    if labels.numel() < num_residues:
        labels = pt.cat([labels, pt.zeros(num_residues - labels.numel())])
        mask = pt.cat([mask, pt.zeros(num_residues - mask.numel())])
    elif labels.numel() > num_residues:
        labels = labels[:num_residues]
        mask = mask[:num_residues]
    receptor_seq = item["receptor_seq_indices"]
    if receptor_seq.numel() < num_residues:
        receptor_seq = pt.cat(
            [receptor_seq, pt.full((num_residues - receptor_seq.numel(),), AA_TO_IDX["X"], dtype=pt.long)]
        )
    elif receptor_seq.numel() > num_residues:
        receptor_seq = receptor_seq[:num_residues]
    partner_seq = sequence_to_tensor(row.get("partner_sequence", ""))
    item.update(
        {
            "y": labels,
            "mask": mask,
            "state_ids": pt.full(
                (num_residues,),
                0 if row.get("input_state") == "holo" else 1,
                dtype=pt.long,
            ),
            "receptor_seq_indices": receptor_seq,
            "receptor_mask": sequence_mask(receptor_seq),
            "partner_seq_indices": partner_seq,
            "partner_mask": sequence_mask(partner_seq),
            "residue_chain_ids": item["residue_chain_ids"][:num_residues],
        }
    )
    return item


def graph_cache_key_for_row(row, ks):
    raw = "|".join([cache_key_for_row(row), ",".join(str(int(k)) for k in ks)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def graph_cache_path_for_row(cache_dir, row, ks):
    return Path(cache_dir) / graph_cache_key_for_row(row, ks)


def build_residue_graph_item(item, ks=(16, 24, 32)):
    """Build the same residue-center kNN payload used by the disk precompute job."""
    X = item["X"].float()
    M = item["M"].float()
    weights = M.to(dtype=X.dtype)
    denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    centers = weights.transpose(0, 1).matmul(X) / denom
    n_res = int(centers.shape[0])
    ks = tuple(int(k) for k in ks)
    graph = {
        "centers": centers.half(),
        "num_residues": n_res,
        "ks": list(ks),
    }
    if n_res <= 1:
        for k in ks:
            graph[f"knn_idx_k{k}"] = pt.empty((n_res, 0), dtype=pt.long)
            graph[f"knn_dist_k{k}"] = pt.empty((n_res, 0), dtype=pt.float16)
        return graph
    distance = pt.cdist(centers, centers).clamp_min(0.0)
    for k in ks:
        effective_k = min(k + 1, n_res)
        neighbor_distance, neighbor_index = pt.topk(
            distance, k=effective_k, dim=-1, largest=False
        )
        graph[f"knn_idx_k{k}"] = neighbor_index[:, 1:].contiguous().long()
        graph[f"knn_dist_k{k}"] = neighbor_distance[:, 1:].contiguous().half()
    return graph


def tensor_tree_nbytes(value):
    if isinstance(value, pt.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    return 0


class BoundedTensorLRU:
    """Process-local tensor LRU. DataLoader workers never write cache files."""

    def __init__(self, max_mb=0):
        self.max_bytes = max(0, int(float(max_mb) * 1024 * 1024))
        self.current_bytes = 0
        self.entries = OrderedDict()

    def get(self, key):
        if key not in self.entries:
            return None
        value, size = self.entries.pop(key)
        self.entries[key] = (value, size)
        return value

    def put(self, key, value):
        if self.max_bytes <= 0:
            return
        size = tensor_tree_nbytes(value)
        if size <= 0 or size > self.max_bytes:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[1]
        while self.entries and self.current_bytes + size > self.max_bytes:
            _old_key, (_old_value, old_size) = self.entries.popitem(last=False)
            self.current_bytes -= old_size
        self.entries[key] = (value, size)
        self.current_bytes += size


def build_apex_tensor_item(row, max_num_nn=64, strict_residue_alignment=True, skip_bad_examples=False):
    structure = read_pdb(row["receptor_pdb_path"])
    aligned_raw_ids = []
    aligned_raw_sequence = []
    if strict_residue_alignment:
        manifest_residue_ids = split_field(row.get("input_residue_ids", ""), str)
        structure, aligned_raw_ids, aligned_raw_sequence = (
            align_polymer_structure_to_residue_ids(structure, manifest_residue_ids)
        )
    structure = clean_structure(structure)
    X, M = encode_structure(structure)
    q = pt.cat(encode_features(structure), dim=1)
    ids_topk = extract_topology(X, int(max_num_nn))[0]
    receptor_seq = structure_sequence_to_tensor(structure)
    residue_chain_ids = structure_residue_chain_ids(structure)

    labels = pt.tensor(split_field(row.get("labels", ""), int), dtype=pt.float)
    mask = pt.tensor(split_field(row.get("label_mask", ""), int), dtype=pt.float)
    n_res = M.shape[1]
    if receptor_seq.numel() < n_res:
        receptor_seq = pt.cat([receptor_seq, pt.full((n_res - receptor_seq.numel(),), AA_TO_IDX["X"], dtype=pt.long)])
    elif receptor_seq.numel() > n_res:
        receptor_seq = receptor_seq[:n_res]
    if strict_residue_alignment:
        if labels.numel() != n_res or mask.numel() != n_res:
            message = (
                f"Residue count mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"labels={labels.numel()} mask={mask.numel()} encoded={n_res}"
            )
            if skip_bad_examples:
                return {
                    "skip": True,
                    "skip_reason": message,
                    "rows": row,
                }
            raise ValueError(message)
        if manifest_residue_ids:
            clean_seq = structure_sequence_list(structure)
        if manifest_residue_ids and (
            manifest_residue_ids != aligned_raw_ids
            or len(aligned_raw_ids) != n_res
            or aligned_raw_sequence != clean_seq
        ):
            message = (
                f"Ordinal residue alignment mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"manifest={len(manifest_residue_ids)} raw={len(aligned_raw_ids)} encoded={n_res}"
            )
            if skip_bad_examples:
                return {
                    "skip": True,
                    "skip_reason": message,
                    "rows": row,
                }
            raise ValueError(message)
    if labels.numel() < n_res:
        labels = pt.cat([labels, pt.zeros(n_res - labels.numel())])
        mask = pt.cat([mask, pt.zeros(n_res - mask.numel())])
    elif labels.numel() > n_res:
        labels = labels[:n_res]
        mask = mask[:n_res]

    partner_seq = sequence_to_tensor(row["partner_sequence"])
    partner_mask = sequence_mask(partner_seq)

    return {
        "X": X,
        "ids_topk": ids_topk,
        "q": q,
        "M": M.float(),
        "y": labels,
        "mask": mask,
        "state_ids": pt.full((n_res,), 0 if row.get("input_state") == "holo" else 1, dtype=pt.long),
        "receptor_seq_indices": receptor_seq,
        "receptor_mask": sequence_mask(receptor_seq),
        "partner_seq_indices": partner_seq,
        "partner_mask": partner_mask,
        "residue_chain_ids": residue_chain_ids[:n_res] if residue_chain_ids.numel() >= n_res else pt.zeros(n_res, dtype=pt.long),
    }


def save_apex_tensor_item(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_item = {k: v.cpu() if isinstance(v, pt.Tensor) else v for k, v in item.items() if k != "rows"}
    pt.save(tensor_item, path)


class ApexPairedManifestDataset(pt.utils.data.Dataset):
    """Paired apo/AF2 receptor + partner-sequence dataset for APEX-Net."""

    def __init__(
        self,
        rows,
        max_num_nn=64,
        strict_residue_alignment=True,
        skip_bad_examples=False,
        tensor_cache_dir=None,
        require_tensor_cache=False,
        compact_structure_cache_dir=None,
        require_compact_structure_cache=False,
        esm_cache_dir=None,
        require_esm_cache=False,
        esm_model_name="esm2_t33_650M_UR50D",
        esm_repr_layer=33,
        esm_max_length=1022,
        residue_graph_cache_dir=None,
        require_residue_graph_cache=False,
        residue_graph_ks=(16, 24, 32),
        build_residue_graph_on_the_fly=False,
        online_memory_cache_mb=0,
        receptor_esm_cache_dir=None,
    ):
        self.rows = list(rows)
        self.max_num_nn = int(max_num_nn)
        self.strict_residue_alignment = bool(strict_residue_alignment)
        self.skip_bad_examples = bool(skip_bad_examples)
        self.tensor_cache_dir = Path(tensor_cache_dir) if tensor_cache_dir else None
        self.require_tensor_cache = bool(require_tensor_cache)
        self.compact_structure_cache_dir = (
            Path(compact_structure_cache_dir) if compact_structure_cache_dir else None
        )
        self.require_compact_structure_cache = bool(require_compact_structure_cache)
        self.esm_cache_dir = Path(esm_cache_dir) if esm_cache_dir else None
        self.require_esm_cache = bool(require_esm_cache)
        self.esm_model_name = str(esm_model_name)
        self.esm_repr_layer = int(esm_repr_layer)
        self.esm_max_length = int(esm_max_length)
        self.receptor_esm_cache_dir = Path(receptor_esm_cache_dir) if receptor_esm_cache_dir else None
        self.residue_graph_cache_dir = Path(residue_graph_cache_dir) if residue_graph_cache_dir else None
        self.require_residue_graph_cache = bool(require_residue_graph_cache)
        self.residue_graph_ks = tuple(int(k) for k in residue_graph_ks)
        self.build_residue_graph_on_the_fly = bool(build_residue_graph_on_the_fly)
        self.online_memory_cache = BoundedTensorLRU(online_memory_cache_mb)

    def _attach_partner_esm_cache(self, item, row):
        if self.esm_cache_dir is None:
            return item
        path = esm_cache_path_for_sequence(
            self.esm_cache_dir,
            row.get("partner_sequence", ""),
            model_name=self.esm_model_name,
            repr_layer=self.esm_repr_layer,
            max_length=self.esm_max_length,
        )
        if not path.is_file():
            message = f"Missing ESM cache for apex_case_id={row.get('apex_case_id', '')}: {path}"
            if self.require_esm_cache or self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        cache = pt.load(path, map_location="cpu")
        emb = cache["embedding"].to(dtype=pt.float)
        item["partner_esm_embedding"] = emb
        item["partner_esm_mask"] = pt.ones((emb.shape[0],), dtype=pt.float)
        return item

    def _attach_receptor_esm_cache(self, item, row):
        if self.receptor_esm_cache_dir is None:
            return item
        cid = row.get("apex_case_id", "").strip()
        if not cid:
            return item
        # Load hash mapping (cached in memory)
        if not hasattr(self, "_receptor_esm_hash_map"):
            map_file = self.receptor_esm_cache_dir / "case_id_map.json"
            if map_file.is_file():
                import json
                with open(map_file) as f:
                    self._receptor_esm_hash_map = json.load(f)
            else:
                self._receptor_esm_hash_map = {}
        hash_id = self._receptor_esm_hash_map.get(cid)
        if not hash_id:
            return item
        path = self.receptor_esm_cache_dir / f"{hash_id}.pt"
        if not path.is_file():
            return item
        try:
            emb = pt.load(path, map_location="cpu")
            if emb.dtype != pt.float32:
                emb = emb.to(dtype=pt.float32)
            item["receptor_esm_embedding"] = emb
            item["receptor_esm_mask"] = pt.ones((emb.shape[0],), dtype=pt.float)
        except Exception:
            pass
        return item

    def _attach_residue_graph_cache(self, item, row):
        if "residue_graph_cache" in item:
            cache = item["residue_graph_cache"]
            n_res = int(item["M"].shape[1])
            if int(cache.get("num_residues", -1)) != n_res:
                message = (
                    f"Embedded residue graph size mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                    f"cache={cache.get('num_residues')} tensor={n_res}"
                )
                if self.require_residue_graph_cache or self.skip_bad_examples:
                    return {"skip": True, "skip_reason": message, "rows": row}
                item.pop("residue_graph_cache", None)
            else:
                return item
        key = graph_cache_key_for_row(row, self.residue_graph_ks)
        cache = self.online_memory_cache.get(("graph", key))
        path = None
        if cache is None and self.residue_graph_cache_dir is not None:
            path = graph_cache_path_for_row(self.residue_graph_cache_dir, row, self.residue_graph_ks)
            if path.is_file():
                cache = pt.load(path, map_location="cpu")
        if cache is None and self.build_residue_graph_on_the_fly:
            cache = build_residue_graph_item(item, self.residue_graph_ks)
            self.online_memory_cache.put(("graph", key), cache)
        if cache is None:
            message = f"Missing residue graph cache for apex_case_id={row.get('apex_case_id', '')}: {path}"
            if self.require_residue_graph_cache or self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        n_res = int(item["M"].shape[1])
        if int(cache.get("num_residues", -1)) != n_res:
            message = (
                f"Residue graph cache size mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"cache={cache.get('num_residues')} tensor={n_res}"
            )
            if self.require_residue_graph_cache or self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        item["residue_graph_cache"] = cache
        return item

    def _attach_residue_chain_ids(self, item, row):
        if item.get("skip") or "residue_chain_ids" in item:
            return item
        n_res = int(item["M"].shape[1])
        chain_ids = residue_chain_ids_for_pdb(row.get("receptor_pdb_path", ""))
        if isinstance(chain_ids, pt.Tensor) and int(chain_ids.numel()) >= n_res:
            item["residue_chain_ids"] = chain_ids[:n_res].long()
        else:
            item["residue_chain_ids"] = pt.zeros(n_res, dtype=pt.long)
        return item

    def _attach_optional_caches(self, item, row):
        item = self._attach_residue_chain_ids(item, row)
        if item.get("skip"):
            return item
        item = self._attach_partner_esm_cache(item, row)
        if item.get("skip"):
            return item
        item = self._attach_receptor_esm_cache(item, row)
        if item.get("skip"):
            return item
        item = self._attach_residue_graph_cache(item, row)
        return item

    def _attach_manifest_labels(self, item, row):
        if str(row.get("tensor_cache_labels_from_manifest", "")).lower() not in {"1", "true", "yes"}:
            return item
        labels = pt.tensor(split_field(row.get("labels", ""), int), dtype=pt.float)
        mask = pt.tensor(split_field(row.get("label_mask", ""), int), dtype=pt.float)
        num_residues = int(item["M"].shape[1])
        if labels.numel() != num_residues or mask.numel() != num_residues:
            message = (
                f"Manifest label overlay mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"labels={labels.numel()} mask={mask.numel()} encoded={num_residues}"
            )
            if self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            raise ValueError(message)
        item["y"] = labels
        item["mask"] = mask
        partner_seq = sequence_to_tensor(row.get("partner_sequence", ""))
        item["partner_seq_indices"] = partner_seq
        item["partner_mask"] = sequence_mask(partner_seq)
        item["state_ids"] = pt.full(
            (num_residues,),
            0 if row.get("input_state") == "holo" else 1,
            dtype=pt.long,
        )
        return item

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tensor_key = cache_key_for_row(row)
        memory_item = self.online_memory_cache.get(("tensor", tensor_key))
        if memory_item is not None:
            item = dict(memory_item)
            item = self._attach_manifest_labels(item, row)
            if not item.get("skip"):
                item["rows"] = row
                item = self._attach_optional_caches(item, row)
            return item
        if self.tensor_cache_dir is not None:
            cache_path = cache_path_for_row(self.tensor_cache_dir, row)
            if cache_path.is_file():
                try:
                    item = pt.load(cache_path, map_location="cpu")
                except Exception as exc:
                    message = (
                        f"Failed to load tensor cache for apex_case_id={row.get('apex_case_id', '')}: "
                        f"{cache_path} ({type(exc).__name__}: {exc})"
                    )
                    if self.skip_bad_examples:
                        return {
                            "skip": True,
                            "skip_reason": message,
                            "rows": row,
                        }
                    raise
                item = self._attach_manifest_labels(item, row)
                if item.get("skip"):
                    return item
                item["rows"] = row
                item = self._attach_optional_caches(item, row)
                return item
            if self.require_tensor_cache:
                message = (
                    f"Missing tensor cache for apex_case_id={row.get('apex_case_id', '')}: {cache_path}"
                )
                if self.skip_bad_examples:
                    return {
                        "skip": True,
                        "skip_reason": message,
                        "rows": row,
                    }
                raise ValueError(message)
        if self.compact_structure_cache_dir is not None:
            compact_path = compact_structure_cache_path_for_row(
                self.compact_structure_cache_dir,
                row,
            )
            if compact_path.is_file():
                try:
                    item = load_compact_structure_npz(
                        compact_path,
                        row,
                        max_num_nn=self.max_num_nn,
                        strict_residue_alignment=self.strict_residue_alignment,
                    )
                except Exception as exc:
                    message = (
                        f"Failed to load compact structure for apex_case_id={row.get('apex_case_id', '')}: "
                        f"{compact_path} ({type(exc).__name__}: {exc})"
                    )
                    if self.skip_bad_examples:
                        return {"skip": True, "skip_reason": message, "rows": row}
                    raise
                self.online_memory_cache.put(("tensor", tensor_key), dict(item))
                item["rows"] = row
                item = self._attach_optional_caches(item, row)
                return item
            if self.require_compact_structure_cache:
                message = (
                    f"Missing compact structure for apex_case_id={row.get('apex_case_id', '')}: "
                    f"{compact_path}"
                )
                if self.skip_bad_examples:
                    return {"skip": True, "skip_reason": message, "rows": row}
                raise ValueError(message)
        item = build_apex_tensor_item(
            row,
            max_num_nn=self.max_num_nn,
            strict_residue_alignment=self.strict_residue_alignment,
            skip_bad_examples=self.skip_bad_examples,
        )
        if not item.get("skip"):
            self.online_memory_cache.put(("tensor", tensor_key), dict(item))
        if not item.get("skip"):
            item = self._attach_manifest_labels(item, row)
        if not item.get("skip"):
            item["rows"] = row
            item = self._attach_optional_caches(item, row)
        return item


def collate_apex_batch(batch):
    skipped = [item for item in batch if item.get("skip")]
    batch = [item for item in batch if not item.get("skip")]
    if not batch:
        return {
            "skip_batch": True,
            "skipped_count": len(skipped),
            "skipped_rows": [item.get("rows", {}) for item in skipped],
            "skip_reasons": [item.get("skip_reason", "") for item in skipped],
        }
    X, ids_topk, q, M = collate_batch_features(
        [[item["X"], item["ids_topk"], item["q"], item["M"]] for item in batch]
    )
    atom_counts = pt.tensor([item["X"].shape[0] for item in batch], dtype=pt.long)
    residue_counts = pt.tensor([item["M"].shape[1] for item in batch], dtype=pt.long)
    y = pt.cat([item["y"] for item in batch], dim=0)
    mask = pt.cat([item["mask"] for item in batch], dim=0)
    state_ids = pt.cat([item["state_ids"] for item in batch], dim=0)
    residue_chain_ids = pt.cat([item.get("residue_chain_ids", pt.zeros(item["M"].shape[1], dtype=pt.long)) for item in batch], dim=0)
    rows = [item["rows"] for item in batch]

    receptor_seq = pt.nn.utils.rnn.pad_sequence(
        [item["receptor_seq_indices"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    receptor_mask = pt.nn.utils.rnn.pad_sequence(
        [item["receptor_mask"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    partner_seq = pt.nn.utils.rnn.pad_sequence(
        [item["partner_seq_indices"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    partner_mask = pt.nn.utils.rnn.pad_sequence(
        [item["partner_mask"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    out = {
        "X": X,
        "ids_topk": ids_topk,
        "q": q,
        "M": M,
        "atom_counts": atom_counts,
        "residue_counts": residue_counts,
        "y": y,
        "mask": mask,
        "state_ids": state_ids,
        "residue_chain_ids": residue_chain_ids,
        "receptor_seq_indices": receptor_seq,
        "receptor_mask": receptor_mask,
        "partner_seq_indices": partner_seq,
        "partner_mask": partner_mask,
        "rows": rows,
        "skip_batch": False,
        "skipped_count": len(skipped),
        "skipped_rows": [item.get("rows", {}) for item in skipped],
        "skip_reasons": [item.get("skip_reason", "") for item in skipped],
    }
    if all("receptor_esm_embedding" in item for item in batch):
        out["receptor_esm_embedding"] = pt.nn.utils.rnn.pad_sequence(
            [item["receptor_esm_embedding"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    if any("partner_esm_embedding" in item for item in batch):
        if not all("partner_esm_embedding" in item for item in batch):
            raise ValueError("Mixed batch with partial partner_esm_embedding cache is not supported.")
        out["partner_esm_embedding"] = pt.nn.utils.rnn.pad_sequence(
            [item["partner_esm_embedding"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        out["partner_esm_mask"] = pt.nn.utils.rnn.pad_sequence(
            [item["partner_esm_mask"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    if any("residue_graph_cache" in item for item in batch):
        if not all("residue_graph_cache" in item for item in batch):
            raise ValueError("Mixed batch with partial residue_graph_cache is not supported.")
        graph_caches = [item["residue_graph_cache"] for item in batch]
        out["residue_graph_caches"] = graph_caches
    return out
