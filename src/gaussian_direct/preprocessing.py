#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import torch as pt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .pesto.dataset import (
    build_apex_tensor_item,
    build_residue_graph_item,
    compact_structure_cache_key_for_row,
    compact_structure_cache_path_for_row,
    compact_structure_identity_for_row,
    load_compact_structure_npz,
    save_compact_structure_npz,
)
from .pesto.surface_cache import build_s1_surface_cache_for_version


def configure_torch_threads(num_threads):
    pt.set_num_threads(max(1, int(num_threads)))
    try:
        pt.set_num_interop_threads(1)
    except RuntimeError:
        pass


def unique_rows(path):
    rows = {}
    conflicts = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = compact_structure_cache_key_for_row(row)
            previous = rows.get(key)
            if previous is None:
                rows[key] = row
                continue
            fields = ("receptor_pdb_path", "receptor_pdb_relpath", "input_residue_ids")
            if any(str(previous.get(name, "")) != str(row.get(name, "")) for name in fields):
                conflicts.append(
                    {
                        "key": key,
                        "identity": compact_structure_identity_for_row(row),
                        "first_case_id": previous.get("apex_case_id", ""),
                        "second_case_id": row.get("apex_case_id", ""),
                    }
                )
    return list(rows.values()), conflicts


def worker(payload):
    (
        index,
        row,
        output_dir,
        max_num_nn,
        overwrite,
        strict_alignment,
        payload_mode,
        torch_threads,
    ) = payload
    configure_torch_threads(torch_threads)
    path = compact_structure_cache_path_for_row(output_dir, row)
    case_id = row.get("apex_case_id", "")
    try:
        if path.is_file() and path.stat().st_size > 0 and not overwrite:
            try:
                load_compact_structure_npz(
                    path,
                    row,
                    max_num_nn=max_num_nn,
                    strict_residue_alignment=strict_alignment,
                )
                return {
                    "index": index,
                    "ok": True,
                    "status": "exists",
                    "bytes": path.stat().st_size,
                }
            except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                # An overlay cache may contain a payload built for an older
                # residue contract. Rebuild it instead of treating it as final.
                pass
        item = build_apex_tensor_item(
            row,
            max_num_nn=max_num_nn,
            strict_residue_alignment=strict_alignment,
            skip_bad_examples=False,
        )
        graph = None
        surface = None
        if payload_mode == "training":
            graph = build_residue_graph_item(item, ks=(16, 24, 32))
            surface = build_s1_surface_cache_for_version(
                item,
                version="s1_surface_shrake_v1",
                num_surface_points=1024,
                graph_k=16,
                lappe_dim=0,
                shrake_probe_radius=1.4,
                shrake_sphere_samples=64,
                shrake_neighbor_method="bruteforce",
            )
        save_compact_structure_npz(
            path,
            item,
            row,
            residue_graph_cache=graph,
            s1_surface_cache=surface,
        )
        replay = load_compact_structure_npz(
            path,
            row,
            max_num_nn=max_num_nn,
            strict_residue_alignment=strict_alignment,
        )
        exact_keys = (
            "X",
            "ids_topk",
            "q",
            "M",
            "receptor_seq_indices",
            "residue_chain_ids",
        )
        non_exact = [name for name in exact_keys if not pt.equal(item[name], replay[name])]
        if non_exact:
            raise ValueError(f"compact_replay_non_exact:{','.join(non_exact)}")
        if graph is not None:
            for name, value in graph.items():
                candidate = replay["residue_graph_cache"].get(name)
                if isinstance(value, pt.Tensor):
                    if not pt.equal(value, candidate):
                        raise ValueError(f"compact_graph_replay_non_exact:{name}")
                elif value != candidate:
                    raise ValueError(f"compact_graph_replay_non_exact:{name}")
        if surface is not None:
            for name, value in surface.items():
                if name == "proxy_surface_labels":
                    continue
                candidate = replay["s1_surface_cache"].get(name)
                if isinstance(value, pt.Tensor):
                    if not pt.equal(value, candidate):
                        raise ValueError(f"compact_surface_replay_non_exact:{name}")
                elif value != candidate:
                    raise ValueError(f"compact_surface_replay_non_exact:{name}")
        return {
            "index": index,
            "ok": True,
            "status": "written",
            "bytes": path.stat().st_size,
            "num_atoms": int(item["X"].shape[0]),
            "num_residues": int(item["M"].shape[1]),
        }
    except Exception as exc:
        return {
            "index": index,
            "ok": False,
            "status": "error",
            "case_id": case_id,
            "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Build one compact, lossless NPZ per unique receptor structure."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-num-nn", type=int, default=64)
    parser.add_argument("--payload", choices=["base", "training"], default="training")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument(
        "--trust-final-filter",
        action="store_true",
        help="Skip the raw residue-order re-audit for a manifest already accepted by the legacy final filter.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, conflicts = unique_rows(args.manifest)
    if conflicts:
        (output_dir / "identity_conflicts.json").write_text(
            json.dumps(conflicts[:1000], indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Found {len(conflicts)} compact structure identity conflicts.")
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No unique structures found in manifest.")

    tasks = [
        (
            index,
            row,
            str(output_dir),
            args.max_num_nn,
            args.overwrite,
            not args.trust_final_filter,
            args.payload,
            args.torch_threads,
        )
        for index, row in enumerate(rows)
    ]
    start = time.time()
    results = []
    iterator = map(worker, tasks)
    if args.workers > 1:
        pool = Pool(processes=args.workers)
        iterator = pool.imap_unordered(worker, tasks, chunksize=max(1, args.chunksize))
    else:
        pool = None
    try:
        for completed, result in enumerate(iterator, 1):
            results.append(result)
            if completed == 1 or completed % args.progress_every == 0:
                print(
                    f"compact_npz {completed}/{len(tasks)} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    counts = Counter(result["status"] for result in results)
    failures = [result for result in results if not result["ok"]]
    report = {
        "schema_version": "m4_compact_structure_npz_v2",
        "payload": args.payload,
        "manifest": str(Path(args.manifest).resolve()),
        "unique_structures": len(rows),
        "ok_structures": len(rows) - len(failures),
        "bad_structures": len(failures),
        "status_counts": dict(counts),
        "output_bytes": sum(int(result.get("bytes", 0)) for result in results),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (output_dir / "compact_structure_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    failure_path = output_dir / "compact_structure_failures.jsonl"
    if failures:
        with failure_path.open("w", encoding="utf-8") as handle:
            for result in sorted(failures, key=lambda value: value["index"]):
                handle.write(json.dumps(result) + "\n")
    elif failure_path.exists():
        failure_path.unlink()
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
