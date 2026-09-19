"""Multi-view residue graph backbone used by M4."""

import math

import torch as pt

from .gvp import FullGVPMessageLayer, GVP, GVPDropout, GVPTupleLayerNorm, masked_mean


def parse_int_tuple(text_or_values):
    if isinstance(text_or_values, str):
        return tuple(int(x) for x in text_or_values.split(",") if str(x).strip())
    return tuple(int(x) for x in text_or_values)


class ProteinTokenEncoder(pt.nn.Module):
    """Lightweight sequence-context encoder fused into residue scalar input."""

    def __init__(self, input_dim, dim=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(int(input_dim)),
            pt.nn.Linear(int(input_dim), dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.conv_norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(int(layers))])
        self.conv = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim),
                    pt.nn.GELU(),
                    pt.nn.Conv1d(dim, dim, kernel_size=1),
                    pt.nn.Dropout(dropout),
                )
                for _ in range(int(layers))
            ]
        )
        encoder_layer = pt.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=max(1, int(heads)),
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = pt.nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        self.out_norm = pt.nn.LayerNorm(dim)

    def forward(self, token_features):
        h = self.input_proj(token_features).unsqueeze(0)
        for norm, block in zip(self.conv_norms, self.conv):
            update = block(norm(h).transpose(1, 2)).transpose(1, 2)
            h = h + update
        h = self.transformer(h)
        return self.out_norm(h.squeeze(0))


class ResidueFeatureBuilder(pt.nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        dim,
        vector_dim,
        rbf_bins=16,
        dropout=0.1,
        extra_residue_feature_dim=0,
        token_encoder_mode="none",
        token_encoder_layers=2,
        token_encoder_heads=4,
        sequence_pos_enc_mode="none",
        sequence_pos_enc_dim=16,
    ):
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.dim = int(dim)
        self.vector_dim = int(vector_dim)
        self.extra_residue_feature_dim = int(extra_residue_feature_dim)
        centers = pt.linspace(0.0, 30.0, int(rbf_bins))
        self.register_buffer("rbf_centers", centers)
        self.node_scalar_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(atom_feature_dim + 6),
            pt.nn.Linear(atom_feature_dim + 6, dim),
            pt.nn.SiLU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        if self.extra_residue_feature_dim > 0:
            self.extra_residue_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(self.extra_residue_feature_dim),
                pt.nn.Linear(self.extra_residue_feature_dim, dim),
                pt.nn.SiLU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
        else:
            self.extra_residue_proj = None
        self.token_encoder_mode = str(token_encoder_mode)
        if self.token_encoder_mode not in {"none", "protein_token"}:
            raise ValueError("token_encoder_mode must be one of: none, protein_token")
        self.sequence_pos_enc_mode = str(sequence_pos_enc_mode)
        if self.sequence_pos_enc_mode not in {"none", "rel_bucket"}:
            raise ValueError("sequence_pos_enc_mode must be one of: none, rel_bucket")
        self.sequence_pos_enc_dim = max(1, int(sequence_pos_enc_dim))
        if self.sequence_pos_enc_mode != "none":
            self.register_buffer("sequence_pos_bucket_bounds", pt.tensor([0, 1, 2, 3, 4, 8, 16, 32, 64], dtype=pt.long))
            self.sequence_pos_bucket_embedding = pt.nn.Embedding(10, self.sequence_pos_enc_dim)
            self.sequence_pos_sign_embedding = pt.nn.Embedding(3, self.sequence_pos_enc_dim)
        else:
            self.sequence_pos_bucket_bounds = None
            self.sequence_pos_bucket_embedding = None
            self.sequence_pos_sign_embedding = None
        self.token_encoder = None
        self.token_gate = None
        if self.token_encoder_mode != "none":
            self.token_encoder = ProteinTokenEncoder(
                atom_feature_dim + 6,
                dim=dim,
                layers=token_encoder_layers,
                heads=token_encoder_heads,
                dropout=dropout,
            )
            self.token_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
                pt.nn.Sigmoid(),
            )
            self.token_fusion_norm = pt.nn.LayerNorm(dim)
        self.last_diagnostics = {}
        self.node_vector_proj = pt.nn.Linear(4, vector_dim, bias=False)
        self.sidechain_vector_proj = pt.nn.Linear(1, vector_dim, bias=False)
        edge_scalar_dim = int(rbf_bins) + 2
        if self.sequence_pos_enc_mode != "none":
            edge_scalar_dim += self.sequence_pos_enc_dim
        self.edge_scalar_proj = pt.nn.Sequential(
            pt.nn.Linear(edge_scalar_dim, dim),
            pt.nn.SiLU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.edge_vector_proj = pt.nn.Linear(1, vector_dim, bias=False)
        self.input_norm = GVPTupleLayerNorm(dim, vector_dim)

    @staticmethod
    def _scatter_sum(values, atom_to_residue_index, num_residues):
        out = values.new_zeros((int(num_residues),) + tuple(values.shape[1:]))
        out.index_add_(0, atom_to_residue_index.to(device=values.device, dtype=pt.long), values)
        return out

    @staticmethod
    def _assignment(M, atom_to_residue_index, num_residues, device):
        if atom_to_residue_index is not None:
            if num_residues is None:
                raise ValueError("num_residues is required with atom_to_residue_index.")
            return atom_to_residue_index.to(device=device, dtype=pt.long), int(num_residues)
        if M is None:
            raise ValueError("Either M or atom_to_residue_index must be provided.")
        return M.to(device=device).argmax(dim=1).long(), int(M.shape[1])

    def centers_from_atom_mask(self, X, M=None, atom_to_residue_index=None, num_residues=None):
        if M is not None and atom_to_residue_index is None:
            weights = M.to(dtype=X.dtype, device=X.device)
            denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            return weights.transpose(0, 1).matmul(X) / denom
        atom_index, residue_count = self._assignment(M, atom_to_residue_index, num_residues, X.device)
        sums = self._scatter_sum(X, atom_index, residue_count)
        counts = self._scatter_sum(X.new_ones((X.shape[0], 1)), atom_index, residue_count).clamp_min(1.0)
        return sums / counts

    def rbf(self, dist):
        width = (self.rbf_centers[1] - self.rbf_centers[0]).abs().clamp_min(1e-3)
        return pt.exp(-((dist.unsqueeze(-1) - self.rbf_centers.to(dist.device)) / width) ** 2)

    def aggregate_atom_features(self, q_atom, M=None, atom_to_residue_index=None, num_residues=None):
        if M is not None and atom_to_residue_index is None:
            weights = M.to(dtype=q_atom.dtype, device=q_atom.device)
            denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            mean_feat = weights.transpose(0, 1).matmul(q_atom) / denom
            atom_count = weights.sum(dim=0).log1p().unsqueeze(-1)
            has_atoms = (weights.sum(dim=0) > 0).to(dtype=q_atom.dtype).unsqueeze(-1)
            atom_offset = self.atom_feature_dim - 64
            if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
                bb = weights.transpose(0, 1).matmul(q_atom[:, atom_offset : atom_offset + 4]).clamp_max(1.0)
            else:
                bb = q_atom.new_zeros(M.shape[1], 4)
            bb_quality = bb.mean(dim=-1, keepdim=True)
            return pt.cat([mean_feat, atom_count, has_atoms, bb_quality, bb[:, :3]], dim=-1)
        atom_index, residue_count = self._assignment(M, atom_to_residue_index, num_residues, q_atom.device)
        counts = self._scatter_sum(q_atom.new_ones((q_atom.shape[0], 1)), atom_index, residue_count)
        mean_feat = self._scatter_sum(q_atom, atom_index, residue_count) / counts.clamp_min(1.0)
        atom_count = counts.log1p()
        has_atoms = (counts > 0).to(dtype=q_atom.dtype)
        atom_offset = self.atom_feature_dim - 64
        if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
            bb = self._scatter_sum(q_atom[:, atom_offset : atom_offset + 4], atom_index, residue_count).clamp_max(1.0)
        else:
            bb = q_atom.new_zeros(residue_count, 4)
        bb_quality = bb.mean(dim=-1, keepdim=True)
        return pt.cat([mean_feat, atom_count, has_atoms, bb_quality, bb[:, :3]], dim=-1)

    def backbone_vectors(self, X, M, q_atom, atom_to_residue_index=None, num_residues=None):
        if M is not None and atom_to_residue_index is None:
            centers = self.centers_from_atom_mask(X, M)
            n_res = M.shape[1]
            atom_offset = self.atom_feature_dim - 64
            vecs = []
            if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
                for j in range(4):
                    atom_w = M.to(dtype=X.dtype, device=X.device) * q_atom[:, atom_offset + j : atom_offset + j + 1].to(
                        dtype=X.dtype
                    )
                    denom = atom_w.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
                    pos = atom_w.transpose(0, 1).matmul(X) / denom
                    v = pos - centers
                    present = (atom_w.sum(dim=0) > 0).to(dtype=X.dtype).unsqueeze(-1)
                    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    vecs.append(v * present)
            if not vecs:
                vecs = [X.new_zeros(n_res, 3) for _ in range(4)]
            return pt.stack(vecs, dim=-2), centers
        atom_index, n_res = self._assignment(M, atom_to_residue_index, num_residues, X.device)
        centers = self.centers_from_atom_mask(X, atom_to_residue_index=atom_index, num_residues=n_res)
        atom_offset = self.atom_feature_dim - 64
        vecs = []
        if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
            for j in range(4):
                role = q_atom[:, atom_offset + j : atom_offset + j + 1].to(dtype=X.dtype, device=X.device)
                denom = self._scatter_sum(role, atom_index, n_res).clamp_min(1.0)
                pos = self._scatter_sum(role * X, atom_index, n_res) / denom
                v = pos - centers
                present = (self._scatter_sum(role, atom_index, n_res) > 0).to(dtype=X.dtype)
                v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                vecs.append(v * present)
        if not vecs:
            vecs = [X.new_zeros(n_res, 3) for _ in range(4)]
        return pt.stack(vecs, dim=-2), centers

    def backbone_atom_positions(self, X, M, q_atom, atom_to_residue_index=None, num_residues=None):
        if M is not None and atom_to_residue_index is None:
            centers = self.centers_from_atom_mask(X, M)
            n_res = M.shape[1]
            atom_offset = self.atom_feature_dim - 64
            positions = []
            present_flags = []
            if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
                for j in range(4):
                    atom_w = M.to(dtype=X.dtype, device=X.device) * q_atom[:, atom_offset + j : atom_offset + j + 1].to(
                        dtype=X.dtype
                    )
                    denom = atom_w.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
                    pos = atom_w.transpose(0, 1).matmul(X) / denom
                    present = (atom_w.sum(dim=0) > 0).to(dtype=X.dtype).unsqueeze(-1)
                    positions.append(pt.where(present > 0, pos, centers))
                    present_flags.append(present)
            if not positions:
                positions = [centers for _ in range(4)]
                present_flags = [X.new_zeros(n_res, 1) for _ in range(4)]
            return pt.stack(positions, dim=-2), pt.cat(present_flags, dim=-1), centers
        atom_index, n_res = self._assignment(M, atom_to_residue_index, num_residues, X.device)
        centers = self.centers_from_atom_mask(X, atom_to_residue_index=atom_index, num_residues=n_res)
        atom_offset = self.atom_feature_dim - 64
        positions = []
        present_flags = []
        if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
            for j in range(4):
                role = q_atom[:, atom_offset + j : atom_offset + j + 1].to(dtype=X.dtype, device=X.device)
                role_sum = self._scatter_sum(role, atom_index, n_res)
                pos = self._scatter_sum(role * X, atom_index, n_res) / role_sum.clamp_min(1.0)
                present = (role_sum > 0).to(dtype=X.dtype)
                positions.append(pt.where(present > 0, pos, centers))
                present_flags.append(present)
        if not positions:
            positions = [centers for _ in range(4)]
            present_flags = [X.new_zeros(n_res, 1) for _ in range(4)]
        return pt.stack(positions, dim=-2), pt.cat(present_flags, dim=-1), centers

    def backbone_frames(self, X, M, q_atom, atom_to_residue_index=None, num_residues=None):
        positions, present, centers = self.backbone_atom_positions(
            X, M, q_atom, atom_to_residue_index=atom_to_residue_index, num_residues=num_residues
        )
        # PeSTo atom-name one-hots follow std_names: CA, N, C, O, ...
        pos_ca = positions[:, 0]
        pos_n = positions[:, 1]
        pos_c = positions[:, 2]
        frame = construct_3d_basis(pos_ca, pos_c, pos_n)
        valid = (present[:, 0] > 0) & (present[:, 1] > 0) & (present[:, 2] > 0)
        eye = pt.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).expand(frame.shape[0], -1, -1)
        frame = pt.where(valid.view(-1, 1, 1), frame, eye)
        return frame, pos_ca, valid, centers

    def cached_knn(self, n, device, graph_cache, k):
        if not graph_cache:
            return None, None
        idx = graph_cache.get(f"knn_idx_k{int(k)}")
        dist = graph_cache.get(f"knn_dist_k{int(k)}")
        if idx is None or dist is None:
            return None, None
        idx = idx.to(device=device, dtype=pt.long)
        dist = dist.to(device=device)
        if idx.shape[0] != n or dist.shape[0] != n:
            return None, None
        max_k = min(int(k), max(n - 1, 0), idx.shape[1], dist.shape[1])
        return idx[:, :max_k].contiguous(), dist[:, :max_k].contiguous()

    def build_knn_edges(self, centers, graph_cache, k):
        n = centers.shape[0]
        if n <= 1:
            idx = pt.empty((n, 0), dtype=pt.long, device=centers.device)
            dist = centers.new_empty((n, 0))
            return idx, dist
        nn_idx, nn_dist = self.cached_knn(n, centers.device, graph_cache, k)
        if nn_idx is not None and nn_dist is not None:
            return nn_idx, nn_dist.to(dtype=centers.dtype)
        dist = pt.cdist(centers, centers).clamp_min(0.0)
        kk = min(int(k) + 1, n)
        nn_dist, nn_idx = pt.topk(dist, k=kk, dim=-1, largest=False)
        return nn_idx[:, 1:].contiguous(), nn_dist[:, 1:].contiguous()

    def _rectangular_edges_from_rows(self, centers, rows):
        n = centers.shape[0]
        width = max((len(row) for row in rows), default=0)
        if n <= 0 or width <= 0:
            idx = pt.empty((n, 0), dtype=pt.long, device=centers.device)
            dist = centers.new_empty((n, 0))
            return idx, dist
        padded = []
        for i, row in enumerate(rows):
            if not row:
                row = [i]
            if len(row) < width:
                row = row + [row[0]] * (width - len(row))
            padded.append(row[:width])
        idx = pt.tensor(padded, dtype=pt.long, device=centers.device)
        dist = (centers[idx] - centers.unsqueeze(1)).norm(dim=-1)
        return idx.contiguous(), dist.contiguous()

    def build_seq_edges(self, centers, offsets=(1,), chain_ids=None):
        n = centers.shape[0]
        if n <= 1:
            idx = pt.empty((n, 0), dtype=pt.long, device=centers.device)
            dist = centers.new_empty((n, 0))
            return idx, dist
        if isinstance(offsets, int):
            offsets = (int(offsets),)
        offsets = tuple(sorted({abs(int(x)) for x in offsets if int(x) != 0}))
        if not offsets:
            offsets = (1,)
        if chain_ids is not None:
            chain_ids = chain_ids.to(device=centers.device).reshape(-1)
            if int(chain_ids.numel()) != int(n):
                raise ValueError(
                    f"chain_ids must contain one value per residue: {chain_ids.numel()} != {n}"
                )
        neighbors = []
        for i in range(n):
            row = []
            for off in offsets:
                j = i - off
                if j >= 0 and (chain_ids is None or chain_ids[j] == chain_ids[i]):
                    row.append(j)
                j = i + off
                if j < n and (chain_ids is None or chain_ids[j] == chain_ids[i]):
                    row.append(j)
            neighbors.append(row)
        return self._rectangular_edges_from_rows(centers, neighbors)

    def build_contact_edges(self, centers, threshold=8.0, max_k=32, exclude_seq_sep=2):
        n = centers.shape[0]
        if n <= 1:
            idx = pt.empty((n, 0), dtype=pt.long, device=centers.device)
            dist = centers.new_empty((n, 0))
            return idx, dist
        dist_mat = pt.cdist(centers, centers).clamp_min(0.0)
        base = pt.arange(n, device=centers.device)
        sep = (base[:, None] - base[None, :]).abs()
        keep = (dist_mat <= float(threshold)) & (sep > int(exclude_seq_sep))
        keep.fill_diagonal_(False)
        score = pt.where(keep, dist_mat, dist_mat.new_full(dist_mat.shape, float("inf")))
        kk = min(int(max_k), max(n - 1, 1))
        nn_dist, nn_idx = pt.topk(score, k=kk, dim=-1, largest=False)
        fallback_dist, fallback_idx = pt.topk(dist_mat.masked_fill(pt.eye(n, device=centers.device, dtype=pt.bool), float("inf")), k=1, dim=-1, largest=False)
        invalid = ~pt.isfinite(nn_dist)
        nn_idx = pt.where(invalid, fallback_idx.expand_as(nn_idx), nn_idx)
        nn_dist = pt.where(invalid, fallback_dist.expand_as(nn_dist), nn_dist)
        return nn_idx.contiguous(), nn_dist.contiguous()

    def build_ss_edges(self, centers, true_feature_cache, k=12):
        if true_feature_cache is None:
            raise ValueError("true_feature_cache is required for secondary-structure graph views.")
        n = centers.shape[0]
        if n <= 1:
            idx = pt.empty((n, 0), dtype=pt.long, device=centers.device)
            dist = centers.new_empty((n, 0))
            return idx, dist
        feat = true_feature_cache.to(device=centers.device, dtype=centers.dtype)
        if feat.shape[0] != n or feat.shape[1] < 3:
            raise ValueError("secondary-structure graph requires true feature cache with at least 3 DSSP/SS columns.")
        ss = feat[:, :3]
        ss_conf, ss_type = ss.max(dim=-1)
        same = (ss_type[:, None] == ss_type[None, :]) & (ss_conf[:, None] > 0.0) & (ss_conf[None, :] > 0.0)
        same.fill_diagonal_(False)
        dist_mat = pt.cdist(centers, centers).clamp_min(0.0)
        score = pt.where(same, dist_mat, dist_mat.new_full(dist_mat.shape, float("inf")))
        kk = min(int(k), max(n - 1, 1))
        nn_dist, nn_idx = pt.topk(score, k=kk, dim=-1, largest=False)
        fallback_dist, fallback_idx = pt.topk(dist_mat.masked_fill(pt.eye(n, device=centers.device, dtype=pt.bool), float("inf")), k=1, dim=-1, largest=False)
        invalid = ~pt.isfinite(nn_dist)
        nn_idx = pt.where(invalid, fallback_idx.expand_as(nn_idx), nn_idx)
        nn_dist = pt.where(invalid, fallback_dist.expand_as(nn_dist), nn_dist)
        return nn_idx.contiguous(), nn_dist.contiguous()

    def build_exposure_edges(self, centers, k=16):
        """Pseudo-exposure graph.

        Boundary-like residues are approximated by distance from the protein
        centroid. Each residue connects to residues with similar radial rank,
        which gives a cheap first version of an exposure/basin view without
        requiring SASA/DSSP preprocessing.
        """
        n = centers.shape[0]
        if n <= 1:
            idx = pt.empty((n, 0), dtype=pt.long, device=centers.device)
            dist = centers.new_empty((n, 0))
            return idx, dist
        radial = (centers - centers.mean(dim=0, keepdim=True)).norm(dim=-1)
        radial_diff = (radial[:, None] - radial[None, :]).abs()
        spatial = pt.cdist(centers, centers).clamp_min(0.0)
        score = radial_diff / radial.std().clamp_min(1e-6) + 0.05 * spatial / spatial.mean().clamp_min(1e-6)
        score.fill_diagonal_(float("inf"))
        kk = min(int(k), max(n - 1, 1))
        _score, idx = pt.topk(score, k=kk, dim=-1, largest=False)
        dist = (centers[idx] - centers.unsqueeze(1)).norm(dim=-1)
        return idx.contiguous(), dist.contiguous()

    def edge_features(self, centers, nn_idx, nn_dist):
        if nn_idx.shape[1] == 0:
            return (
                centers.new_empty((centers.shape[0], 0, self.dim)),
                centers.new_empty((centers.shape[0], 0, self.vector_dim, 3)),
            )
        direction = centers[nn_idx] - centers.unsqueeze(1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        edge_v = self.edge_vector_proj(direction.unsqueeze(-2).transpose(-1, -2)).transpose(-1, -2)
        rel = (nn_idx - pt.arange(centers.shape[0], device=centers.device).unsqueeze(-1)).float()
        rel_abs = pt.log1p(rel.abs()).unsqueeze(-1) / math.log(max(centers.shape[0], 2))
        rel_sign = rel.sign().unsqueeze(-1)
        nn_dist = pt.where(pt.isfinite(nn_dist), nn_dist, nn_dist.new_zeros(nn_dist.shape))
        scalar_parts = [self.rbf(nn_dist.to(dtype=centers.dtype)), rel_abs, rel_sign]
        if self.sequence_pos_enc_mode == "rel_bucket":
            rel_long = (nn_idx - pt.arange(centers.shape[0], device=centers.device).unsqueeze(-1)).to(dtype=pt.long)
            rel_abs_long = rel_long.abs()
            bounds = self.sequence_pos_bucket_bounds.to(device=centers.device)
            bucket = pt.bucketize(rel_abs_long, bounds, right=True).clamp(0, 9)
            sign = rel_long.sign().clamp(-1, 1) + 1
            seq_emb = self.sequence_pos_bucket_embedding(bucket) + self.sequence_pos_sign_embedding(sign)
            scalar_parts.append(seq_emb.to(dtype=centers.dtype))
        edge_s = self.edge_scalar_proj(pt.cat(scalar_parts, dim=-1))
        return edge_s, edge_v

    def node_features(
        self, X, q_atom, M=None, extra_residue_features=None, atom_to_residue_index=None, num_residues=None,
        static_geometry=None,
    ):
        scalar_node = self.aggregate_atom_features(
            q_atom, M, atom_to_residue_index=atom_to_residue_index, num_residues=num_residues
        )
        s = self.node_scalar_proj(scalar_node)
        self.last_diagnostics = {}
        if self.token_encoder is not None:
            s_token = self.token_encoder(scalar_node)
            gate = self.token_gate(pt.cat([s, s_token], dim=-1))
            s = self.token_fusion_norm(s + gate * s_token)
            self.last_diagnostics = {
                "token_encoder_gate_mean": gate.detach().mean(),
                "token_encoder_gate_std": gate.detach().std(unbiased=False),
                "token_encoder_update_norm": (gate * s_token).detach().norm(dim=-1).mean(),
            }
        if self.extra_residue_proj is not None and extra_residue_features is not None:
            extra = extra_residue_features.to(device=s.device, dtype=s.dtype)
            s = s + self.extra_residue_proj(extra)
        if static_geometry is not None:
            centers = static_geometry["centers"].to(device=X.device, dtype=X.dtype)
            base_vec = static_geometry["backbone_vectors"].to(device=X.device, dtype=X.dtype)
        else:
            base_vec, centers = self.backbone_vectors(
                X, M, q_atom, atom_to_residue_index=atom_to_residue_index, num_residues=num_residues
            )
        v = self.node_vector_proj(base_vec.transpose(-1, -2)).transpose(-1, -2)
        return (*self.input_norm(s, v), centers)


class ScalarVector:
    def __init__(self, s, v):
        self.s = s
        self.v = v

    def __add__(self, other):
        if not isinstance(other, ScalarVector):
            return ScalarVector(self.s + other, self.v + other)
        return ScalarVector(self.s + other.s, self.v + other.v)

    def to_tensor(self):
        return pt.cat([self.s, self.v.reshape(*self.v.shape[:-2], -1)], dim=-1)

    @classmethod
    def from_tensor(cls, x, vector_dim):
        vector_size = int(vector_dim) * 3
        s = x[..., : x.shape[-1] - vector_size]
        v = x[..., x.shape[-1] - vector_size :].reshape(*x.shape[:-1], int(vector_dim), 3)
        return cls(s=s, v=v)


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = pt.clamp(pt.sum(pt.square(x), dim=dim, keepdim=keepdim), min=eps)
    return pt.sqrt(out) if sqrt else out


def normalize_vector(v, dim=-1, eps=0.0):
    return v / (pt.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim=-1):
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    e1 = normalize_vector(p1 - center, dim=-1)
    v2 = p2 - center
    e2 = normalize_vector(v2 - project_v2v(v2, e1, dim=-1), dim=-1)
    e3 = pt.cross(e1, e2, dim=-1)
    return pt.stack([e1, e2, e3], dim=-1)


def local_to_global(p, frame):
    p_size = p.shape
    n = p_size[0]
    p_col = p.reshape(n, -1, 3).transpose(-1, -2)
    q = pt.matmul(frame, p_col)
    return q.transpose(-1, -2).reshape(p_size)


def global_to_local(q, frame):
    q_size = q.shape
    n = q_size[0]
    q_col = q.reshape(n, -1, 3).transpose(-1, -2)
    p = pt.matmul(frame.transpose(-1, -2), q_col)
    return p.transpose(-1, -2).reshape(q_size)


class VectorLinear(pt.nn.Module):
    """Official OAGNN directed vector-weight linear map."""

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = int(in_dims)
        self.out_dims = int(out_dims)
        self.weight = pt.nn.Parameter(pt.empty(self.out_dims, self.in_dims, 3))
        self.reset_parameters()

    def reset_parameters(self):
        pt.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _forward_scalar(self, x):
        u, v, w = pt.unbind(self.weight, dim=-1)
        a = pt.nn.functional.linear(x, u, bias=None).unsqueeze(-1)
        b = pt.nn.functional.linear(x, v, bias=None).unsqueeze(-1)
        c = pt.nn.functional.linear(x, w, bias=None).unsqueeze(-1)
        return pt.cat([a, b, c], dim=-1)

    def _forward_vector_dot(self, x):
        u, v, w = pt.unbind(self.weight, dim=-1)
        x0, x1, x2 = pt.unbind(x, dim=-1)
        return (
            pt.nn.functional.linear(x0, u, bias=None)
            + pt.nn.functional.linear(x1, v, bias=None)
            + pt.nn.functional.linear(x2, w, bias=None)
        )

    def _forward_vector_cross(self, x):
        x_exp = x.unsqueeze(-3)
        size = list(x_exp.shape)
        size[-3] = self.out_dims
        x_exp = x_exp.expand(size)
        w_size = ([1] * (x_exp.dim() - 3)) + [self.out_dims, self.in_dims, 3]
        weight = self.weight.reshape(w_size).expand_as(x_exp)
        return pt.cross(weight, x_exp, dim=-1).sum(-2)

    def forward(self, x, input_type, mult_op="dot"):
        if input_type == "scalar":
            return self._forward_scalar(x)
        if input_type == "vector" and mult_op == "dot":
            return self._forward_vector_dot(x)
        if input_type == "vector" and mult_op == "cross":
            return self._forward_vector_cross(x)
        raise ValueError("VectorLinear expects input_type scalar/vector and mult_op dot/cross.")


def vector_input_scalar_linear(linear, x):
    return linear(x.transpose(-1, -2)).transpose(-1, -2)


class SVLinear(pt.nn.Module):
    """Official OAGNN scalar-vector directed linear module."""

    def __init__(self, in_dims, out_dims, scalar_bias=True, hidden_dims=None, share_dot_cross=False):
        super().__init__()
        self.in_s, self.in_v = int(in_dims[0]), int(in_dims[1])
        self.out_s, self.out_v = int(out_dims[0]), int(out_dims[1])
        hidden_dims = hidden_dims if hidden_dims is not None else out_dims
        self.hid_s, self.hid_v = int(hidden_dims[0]), int(hidden_dims[1])
        if self.hid_v == 0:
            self.hid_v = self.in_v
        self.lin_s_s = pt.nn.Linear(self.in_s, self.hid_s, bias=scalar_bias)
        self.lin_s_v = VectorLinear(self.in_s, self.hid_s)
        self.lin_v_s = pt.nn.Linear(self.in_v, self.hid_v, bias=False)
        self.lin_v_dot = VectorLinear(self.in_v, self.hid_v)
        self.lin_v_cro = self.lin_v_dot if share_dot_cross else VectorLinear(self.in_v, self.hid_v)
        self.lin_out_s = pt.nn.Linear(self.hid_s + self.hid_v, self.out_s, bias=scalar_bias) if self.out_s > 0 else None
        self.lin_out_v = pt.nn.Linear(self.hid_s + 2 * self.hid_v, self.out_v, bias=False) if self.out_v > 0 else None

    def forward(self, x):
        s_s_s = self.lin_s_s(x.s)
        s_v_v = self.lin_s_v(x.s, "scalar")
        v_s_v = vector_input_scalar_linear(self.lin_v_s, x.v)
        v_v_v = self.lin_v_cro(x.v, "vector", "cross")
        v_v_s = self.lin_v_dot(x.v, "vector", "dot")
        h_s = pt.cat([s_s_s, v_v_s], dim=-1)
        h_v = pt.cat([s_v_v, v_s_v, v_v_v], dim=-2)
        out_s = self.lin_out_s(h_s) if self.lin_out_s is not None else None
        out_v = vector_input_scalar_linear(self.lin_out_v, h_v) if self.lin_out_v is not None else None
        return ScalarVector(out_s, out_v)


class SVInteraction(pt.nn.Module):
    def __init__(self, scalar_dims, vector_dims):
        super().__init__()
        self.s_to_v = pt.nn.Linear(int(scalar_dims), int(vector_dims))
        self.v_to_s = VectorLinear(int(vector_dims), int(scalar_dims))

    def forward(self, x):
        bias_s = self.v_to_s(x.v, "vector", "dot")
        gate_v = self.s_to_v(x.s).unsqueeze(-1)
        return ScalarVector(x.s + bias_s, x.v * gate_v)


class SVActivation(pt.nn.Module):
    def __init__(self, s_act, v_act):
        super().__init__()
        self.s_act = s_act
        self.v_act = v_act

    @staticmethod
    def _get_scalar_activation(name):
        if name is None:
            return pt.nn.Identity()
        if name == "sigmoid":
            return pt.sigmoid
        return getattr(pt.nn.functional, name)

    @classmethod
    def from_args(cls, scalar_act, vector_act):
        s_act = cls._get_scalar_activation(scalar_act)
        if vector_act is None:
            v_act = pt.nn.Identity()
        elif vector_act[0] == "scale":
            v_act = VectorScaling(cls._get_scalar_activation(vector_act[1]))
        elif vector_act[0] == "project":
            v_act = VectorProjection(vector_act[1])
        else:
            raise ValueError(f"Unknown vector activation class: {vector_act[0]}")
        return cls(s_act, v_act)

    def forward(self, x):
        return ScalarVector(
            self.s_act(x.s) if x.s is not None else None,
            self.v_act(x.v) if x.v is not None else None,
        )


class VectorProjection(pt.nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.vecs = pt.nn.Parameter(pt.randn(int(n_dims), 3), requires_grad=True)
        pt.nn.utils.weight_norm(self, name="vecs", dim=1)

    def forward(self, x):
        dot_prod = (self.vecs_v * x).sum(dim=-1, keepdim=True)
        x_proj = x - dot_prod * self.vecs_v
        return pt.where(dot_prod >= 0, x, x_proj)


class VectorScaling(pt.nn.Module):
    def __init__(self, func=pt.sigmoid):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(safe_norm(x, dim=-1, keepdim=True)) * x


class SVPerceptron(pt.nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        scalar_bias=True,
        hidden_dims=None,
        scalar_act="relu",
        vector_act=("scale", "sigmoid"),
        interaction=True,
    ):
        super().__init__()
        self.linear = SVLinear(in_dims, out_dims, scalar_bias=scalar_bias, hidden_dims=hidden_dims)
        self.interact = SVInteraction(*out_dims) if interaction else pt.nn.Identity()
        if vector_act is not None and vector_act[0] == "project":
            vector_act = ["project", out_dims[1]]
        self.act = SVActivation.from_args(scalar_act, vector_act)

    def forward(self, x):
        x = self.linear(x)
        if x.v is not None:
            x = self.interact(x)
        return self.act(x)


class VectorMLP(pt.nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        n_layers=3,
        scalar_act="relu",
        vector_act=("scale", "sigmoid"),
    ):
        super().__init__()
        layers = []
        if int(n_layers) <= 1:
            layers.append(SVPerceptron(in_dims, out_dims, scalar_act=None, vector_act=None))
        else:
            layers.append(SVPerceptron(in_dims, out_dims, scalar_act=scalar_act, vector_act=vector_act))
            for _ in range(int(n_layers) - 2):
                layers.append(SVPerceptron(out_dims, out_dims, scalar_act=scalar_act, vector_act=vector_act))
            layers.append(SVPerceptron(out_dims, out_dims, scalar_act=None, vector_act=None))
        self.layers = pt.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SVDropout(pt.nn.Module):
    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.scalar_dropout = pt.nn.Dropout(float(drop_rate))
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        s = self.scalar_dropout(x.s)
        v = x.v
        if self.training and self.drop_rate > 0 and v.numel() > 0:
            mask = (pt.rand(*v.shape[:-1], 1, device=v.device) > self.drop_rate).to(dtype=v.dtype)
            v = v * mask / max(1.0 - self.drop_rate, 1e-6)
        return ScalarVector(s, v)


class SVLayerNorm(pt.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s, self.v = int(dims[0]), int(dims[1])
        self.scalar_norm = pt.nn.LayerNorm(self.s)

    def forward(self, x):
        vn = safe_norm(x.v, dim=-1, keepdim=True, sqrt=False)
        vn = pt.sqrt(pt.mean(vn, dim=-2, keepdim=True))
        return ScalarVector(self.scalar_norm(x.s), x.v / vn)


def rotate_apply(layer, x, frame=None):
    if frame is not None:
        x = ScalarVector(x.s, global_to_local(x.v, frame))
    y = layer(x)
    if frame is not None and y.v is not None:
        y = ScalarVector(y.s, local_to_global(y.v, frame))
    return y


class OfficialOAGNNGraphConv(pt.nn.Module):
    """Dense-kNN implementation of the paper/official SVGraphConv message."""

    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_message_layers=3,
        scalar_act="relu",
        vector_act=("scale", "sigmoid"),
        aggr="mean",
    ):
        super().__init__()
        self.in_s, self.in_v = int(in_dims[0]), int(in_dims[1])
        self.out_s, self.out_v = int(out_dims[0]), int(out_dims[1])
        self.edge_s, self.edge_v = int(edge_dims[0]), int(edge_dims[1])
        self.aggr = str(aggr)
        self.message_func = VectorMLP(
            in_dims=(2 * self.in_s + self.edge_s, 2 * self.in_v + self.edge_v),
            out_dims=(self.out_s, self.out_v),
            n_layers=n_message_layers,
            scalar_act=scalar_act,
            vector_act=vector_act,
        )

    def forward(self, x, nn_idx, edge_attr, frame):
        if nn_idx.shape[1] == 0:
            return ScalarVector(
                x.s.new_zeros(x.s.shape[0], self.out_s),
                x.v.new_zeros(x.v.shape[0], self.out_v, 3),
            )
        neigh_s = x.s[nn_idx]
        neigh_v = x.v[nn_idx]
        base_s = x.s.unsqueeze(1).expand_as(neigh_s)
        base_v = x.v.unsqueeze(1).expand_as(neigh_v)
        msg_s = pt.cat([base_s, neigh_s, edge_attr.s], dim=-1)
        msg_v = pt.cat([base_v, neigh_v, edge_attr.v], dim=-2)
        frame_i = frame.unsqueeze(1).expand(-1, nn_idx.shape[1], -1, -1)
        msg_v_local = global_to_local(msg_v.reshape(-1, msg_v.shape[-2], 3), frame_i.reshape(-1, 3, 3)).reshape_as(msg_v)
        message = self.message_func(ScalarVector(msg_s, msg_v_local))
        msg_v_global = local_to_global(message.v.reshape(-1, self.out_v, 3), frame_i.reshape(-1, 3, 3)).reshape_as(message.v)
        if self.aggr == "add":
            out_s = message.s.sum(dim=1)
            out_v = msg_v_global.sum(dim=1)
        else:
            out_s = message.s.mean(dim=1)
            out_v = msg_v_global.mean(dim=1)
        return ScalarVector(out_s, out_v)


class OfficialOAGNNGraphConvLayer(pt.nn.Module):
    """Official OAGNN SVGraphConvLayer adapted to fixed dense view edges."""

    def __init__(self, node_dims, edge_dims, n_message_layers=3, n_ff_layers=2, dropout=0.1, aggr="mean"):
        super().__init__()
        self.conv = OfficialOAGNNGraphConv(
            node_dims,
            node_dims,
            edge_dims,
            n_message_layers=n_message_layers,
            scalar_act="relu",
            vector_act=("scale", "sigmoid"),
            aggr=aggr,
        )
        self.dropout_1 = SVDropout(dropout)
        self.layernorm_1 = SVLayerNorm(node_dims)
        self.ff_func = VectorMLP(
            in_dims=node_dims,
            out_dims=node_dims,
            n_layers=n_ff_layers,
            scalar_act="relu",
            vector_act=("scale", "sigmoid"),
        )
        self.dropout_2 = SVDropout(dropout)
        self.layernorm_2 = SVLayerNorm(node_dims)
        self.last_diagnostics = {}

    def forward(self, s, v, edge_s, edge_v, nn_idx, frame=None, **_kwargs):
        if frame is None:
            frame = pt.eye(3, device=s.device, dtype=s.dtype).unsqueeze(0).expand(s.shape[0], -1, -1)
        frame = frame.to(device=s.device, dtype=s.dtype)
        x = ScalarVector(s=s, v=v)
        edge_attr = ScalarVector(s=edge_s.to(dtype=s.dtype), v=edge_v.to(dtype=v.dtype))
        dh = self.conv(x, nn_idx.to(device=s.device, dtype=pt.long), edge_attr, frame)
        x = self.layernorm_1(x + self.dropout_1(dh))
        dh = rotate_apply(self.ff_func, x, frame)
        out = self.layernorm_2(x + self.dropout_2(dh))
        self.last_diagnostics = {
            "official_oagnn_edge_count_mean": edge_s.new_tensor(float(nn_idx.shape[1])),
            "official_oagnn_message_norm": dh.s.detach().norm(dim=-1).mean(),
            "official_oagnn_vector_norm": out.v.detach().norm(dim=-1).mean(),
        }
        return out.s, out.v


class OAGNNGVPMessageLayer(pt.nn.Module):
    """OAGNN-style orientation-aware message passing inside a GVP layer.

    This replaces the ordinary FullGVPMessageLayer update rather than appending a
    scalar-only refiner after the encoder. Relation embeddings, local frame
    projections and direction/cross-vector channels are part of the edge message
    before the scalar/vector GVP update is applied.
    """

    def __init__(self, node_dims, edge_dims, dropout=0.1, num_relations=8):
        super().__init__()
        ns, nv = node_dims
        es, ev = edge_dims
        self.ns = int(ns)
        self.nv = int(nv)
        self.es = int(es)
        self.ev = int(ev)
        self.num_relations = int(num_relations)
        self.relation_embedding = pt.nn.Embedding(self.num_relations, self.ns)
        self.direction_weight = pt.nn.Parameter(pt.empty(self.num_relations, self.ns, 3))
        self.cross_weight = pt.nn.Parameter(pt.empty(self.num_relations, self.ns, 3))
        edge_scalar_extra = self.ns + 3 + 3 + 3 + 2
        msg_in = (self.ns * 2 + self.es + edge_scalar_extra, self.nv * 2 + self.ev + 2)
        self.msg_gvp1 = GVP(msg_in, (self.ns, self.nv))
        self.msg_norm1 = GVPTupleLayerNorm(self.ns, self.nv)
        self.msg_gvp2 = GVP((self.ns, self.nv), (self.ns, self.nv))
        self.msg_norm2 = GVPTupleLayerNorm(self.ns, self.nv)
        self.attn_mlp = pt.nn.Sequential(
            pt.nn.LayerNorm(msg_in[0]),
            pt.nn.Linear(msg_in[0], self.ns // 2),
            pt.nn.GELU(),
            pt.nn.Linear(self.ns // 2, 1),
        )
        self.dropout = GVPDropout(dropout, dropout)
        self.node_norm = GVPTupleLayerNorm(self.ns, self.nv)
        self.ff_gvp1 = GVP((self.ns, self.nv), (self.ns * 2, self.nv * 2))
        self.ff_gvp2 = GVP((self.ns * 2, self.nv * 2), (self.ns, self.nv), activations=(None, None))
        self.ff_norm = GVPTupleLayerNorm(self.ns, self.nv)
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        pt.nn.init.normal_(self.direction_weight, mean=0.0, std=0.02)
        pt.nn.init.normal_(self.cross_weight, mean=0.0, std=0.02)

    def _safe_normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _local_frames(self, v):
        if v.shape[-2] < 3:
            eye = pt.eye(3, device=v.device, dtype=v.dtype)
            return eye.unsqueeze(0).expand(v.shape[0], -1, -1)
        raw_e1 = v[:, 0]
        fallback_e1 = v.new_zeros(v.shape[0], 3)
        fallback_e1[:, 0] = 1.0
        raw_e1 = pt.where(raw_e1.norm(dim=-1, keepdim=True) > 1e-4, raw_e1, fallback_e1)
        e1 = self._safe_normalize(raw_e1)
        seed = v[:, 1]
        seed_norm = seed.norm(dim=-1, keepdim=True)
        alt = v.new_zeros(v.shape[0], 3)
        alt[:, 2] = 1.0
        seed = pt.where(seed_norm > 1e-4, seed, alt)
        e2 = seed - (seed * e1).sum(dim=-1, keepdim=True) * e1
        alt2 = v.new_zeros(v.shape[0], 3)
        alt2[:, 1] = 1.0
        e2 = pt.where(e2.norm(dim=-1, keepdim=True) > 1e-4, e2, alt2)
        e2 = self._safe_normalize(e2 - (e2 * e1).sum(dim=-1, keepdim=True) * e1)
        e3 = self._safe_normalize(pt.cross(e1, e2, dim=-1))
        return pt.stack([e1, e2, e3], dim=1)

    def _relation_types(self, nn_idx, centers):
        n, e = nn_idx.shape
        if e == 0:
            return centers.new_empty((n, 0), dtype=pt.long)
        base = pt.arange(n, device=nn_idx.device).unsqueeze(-1)
        sep = (nn_idx - base).abs()
        dist = (centers[nn_idx] - centers.unsqueeze(1)).norm(dim=-1)
        rel = pt.full((n, e), self.num_relations - 1, device=nn_idx.device, dtype=pt.long)
        if self.num_relations >= 8:
            rel = pt.where(dist < 18.0, pt.full_like(rel, 6), rel)
            rel = pt.where(dist < 12.0, pt.full_like(rel, 5), rel)
            rel = pt.where(dist < 8.0, pt.full_like(rel, 4), rel)
            rel = pt.where(sep <= 8, pt.minimum(rel, pt.full_like(rel, 3)), rel)
            rel = pt.where(sep <= 4, pt.minimum(rel, pt.full_like(rel, 2)), rel)
            rel = pt.where(sep <= 2, pt.minimum(rel, pt.full_like(rel, 1)), rel)
            rel = pt.where(sep == 1, pt.full_like(rel, 0), rel)
        return rel.clamp(0, self.num_relations - 1)

    def _edge_geometry(self, v, centers, nn_idx):
        frames = self._local_frames(v)
        rel_vec = centers[nn_idx] - centers.unsqueeze(1)
        direction = self._safe_normalize(rel_vec)
        frame_i = frames.unsqueeze(1).expand(-1, nn_idx.shape[1], -1, -1)
        frame_j = frames[nn_idx]
        local_dir = (direction.unsqueeze(-2) * frame_i).sum(dim=-1)
        cross_vec = pt.cross(direction, frame_i[:, :, 0, :], dim=-1)
        cross_local = (cross_vec.unsqueeze(-2) * frame_i).sum(dim=-1)
        orient = (frame_i * frame_j).sum(dim=-1).clamp(-1.0, 1.0)
        sep = nn_idx - pt.arange(centers.shape[0], device=centers.device).unsqueeze(-1)
        rel_abs = pt.log1p(sep.abs().to(dtype=centers.dtype)).unsqueeze(-1) / math.log(max(int(centers.shape[0]), 2))
        rel_sign = sep.sign().to(dtype=centers.dtype).unsqueeze(-1)
        return direction, cross_vec, local_dir, cross_local, orient, rel_abs, rel_sign

    def forward(self, s, v, edge_s, edge_v, nn_idx, centers=None, rel_type=None):
        if nn_idx.shape[1] == 0:
            self.last_diagnostics = {
                "oagnn_edge_count_mean": edge_s.new_tensor(0.0),
                "oagnn_relation_entropy": edge_s.new_tensor(0.0),
                "oagnn_attention_entropy": edge_s.new_tensor(0.0),
                "oagnn_update_norm": edge_s.new_tensor(0.0),
            }
            return s, v
        if centers is None:
            raise ValueError("OAGNNGVPMessageLayer requires residue centers.")
        centers = centers.to(device=s.device, dtype=s.dtype)
        nn_idx = nn_idx.to(device=s.device, dtype=pt.long)
        if rel_type is None:
            rel_type = self._relation_types(nn_idx, centers)
        rel_type = rel_type.to(device=s.device, dtype=pt.long).clamp(0, self.num_relations - 1)
        direction, cross_vec, local_dir, cross_local, orient, rel_abs, rel_sign = self._edge_geometry(v, centers, nn_idx)
        neigh_s = s[nn_idx]
        neigh_v = v[nn_idx]
        base_s = s.unsqueeze(1).expand_as(neigh_s)
        base_v = v.unsqueeze(1).expand_as(neigh_v)
        rel_emb = self.relation_embedding(rel_type).to(dtype=s.dtype)
        dir_w = self.direction_weight[rel_type]
        cross_w = self.cross_weight[rel_type]
        dir_bias = (dir_w * local_dir.unsqueeze(-2)).sum(dim=-1)
        cross_bias = (cross_w * cross_local.unsqueeze(-2)).sum(dim=-1)
        scalar_parts = [
            base_s,
            neigh_s,
            edge_s.to(dtype=s.dtype),
            rel_emb,
            local_dir,
            cross_local,
            orient,
            rel_abs,
            rel_sign,
        ]
        msg_s_in = pt.cat(scalar_parts, dim=-1)
        msg_v_in = pt.cat(
            [
                base_v,
                neigh_v,
                edge_v.to(dtype=v.dtype),
                direction.unsqueeze(-2),
                cross_vec.unsqueeze(-2),
            ],
            dim=-2,
        )
        msg_s, msg_v = self.msg_gvp1(msg_s_in, msg_v_in)
        msg_s, msg_v = self.msg_norm1(msg_s, msg_v)
        msg_s, msg_v = self.msg_gvp2(msg_s + dir_bias + cross_bias, msg_v)
        msg_s, msg_v = self.msg_norm2(msg_s, msg_v)
        attn_logits = self.attn_mlp(msg_s_in).squeeze(-1)
        attn = pt.softmax(attn_logits, dim=-1)
        agg_s = (attn.unsqueeze(-1) * msg_s).sum(dim=1)
        agg_v = (attn.unsqueeze(-1).unsqueeze(-1) * msg_v).sum(dim=1)
        agg_s, agg_v = self.dropout(agg_s, agg_v)
        s, v = self.node_norm(s + agg_s, v + agg_v)
        ff_s, ff_v = self.ff_gvp1(s, v)
        ff_s, ff_v = self.ff_gvp2(ff_s, ff_v)
        ff_s, ff_v = self.dropout(ff_s, ff_v)
        out_s, out_v = self.ff_norm(s + ff_s, v + ff_v)
        rel_onehot = pt.nn.functional.one_hot(rel_type, num_classes=self.num_relations).to(dtype=s.dtype)
        rel_counts = rel_onehot.sum(dim=(0, 1))
        rel_prob = rel_counts / rel_counts.sum().clamp_min(1.0)
        rel_entropy = -(rel_prob * rel_prob.clamp_min(1e-8).log()).sum()
        attn_entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_diagnostics = {
            "oagnn_edge_count_mean": edge_s.new_tensor(float(nn_idx.shape[1])),
            "oagnn_relation_entropy": rel_entropy.detach(),
            "oagnn_attention_entropy": attn_entropy.detach(),
            "oagnn_update_norm": agg_s.detach().norm(dim=-1).mean(),
            "oagnn_direction_weight_norm": self.direction_weight.detach().norm(),
            "oagnn_cross_weight_norm": self.cross_weight.detach().norm(),
            "oagnn_local_dir_std": local_dir.detach().std(unbiased=False),
        }
        return out_s, out_v


class EdgePairEncoderLite(pt.nn.Module):
    """Lightweight persistent edge/pair representation for one graph view."""

    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 5),
            pt.nn.Linear(dim * 5, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.Sigmoid(),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, edge_s, s, nn_idx):
        base_s = s.unsqueeze(1).expand(-1, nn_idx.shape[1], -1)
        neigh_s = s[nn_idx]
        pair_input = pt.cat([edge_s, base_s, neigh_s, (base_s - neigh_s).abs(), base_s * neigh_s], dim=-1)
        update = self.update(pair_input)
        gate = self.gate(pt.cat([edge_s, base_s, neigh_s], dim=-1))
        return self.norm(edge_s + gate * update), update.detach().norm(dim=-1).mean(), gate.detach().mean()


class StaticGraphView(pt.nn.Module):
    def __init__(
        self,
        dim,
        vector_dim,
        num_layers=2,
        dropout=0.1,
        gear_view_mode="none",
        num_relations=8,
        oagnn_mode="none",
        edge_pair_mode="none",
    ):
        super().__init__()
        self.gear_view_mode = str(gear_view_mode)
        self.oagnn_mode = str(oagnn_mode)
        if self.oagnn_mode not in {"none", "oa_edge", "official"}:
            raise ValueError("oagnn_mode must be one of: none, oa_edge, official")
        self.edge_pair_mode = str(edge_pair_mode)
        if self.edge_pair_mode not in {"none", "lite"}:
            raise ValueError("edge_pair_mode must be one of: none, lite")
        self.num_relations = int(num_relations)
        layers = []
        for _ in range(int(num_layers)):
            if self.oagnn_mode == "official":
                layers.append(OfficialOAGNNGraphConvLayer((dim, vector_dim), (dim, vector_dim), dropout=dropout))
            elif self.oagnn_mode == "oa_edge":
                layers.append(OAGNNGVPMessageLayer((dim, vector_dim), (dim, vector_dim), dropout=dropout, num_relations=self.num_relations))
            else:
                layers.append(FullGVPMessageLayer((dim, vector_dim), (dim, vector_dim), dropout=dropout))
        self.layers = pt.nn.ModuleList(layers)
        self.edge_updates = pt.nn.ModuleList()
        self.edge_norms = pt.nn.ModuleList()
        self.pair_updates = pt.nn.ModuleList()
        self.relation_embedding = None
        self.dropout = pt.nn.Dropout(dropout)
        if self.edge_pair_mode != "none":
            self.pair_updates = pt.nn.ModuleList([EdgePairEncoderLite(dim=dim, dropout=dropout) for _ in range(int(num_layers))])
        if self.gear_view_mode != "none":
            if self.gear_view_mode not in {"rel_edge_msg", "rel_edge_msg_ieconv"}:
                raise ValueError("gear_view_mode must be one of: none, rel_edge_msg, rel_edge_msg_ieconv")
            self.relation_embedding = pt.nn.Embedding(self.num_relations, dim)
            self.edge_updates = pt.nn.ModuleList(
                [
                    pt.nn.Sequential(
                        pt.nn.LayerNorm(dim * 4),
                        pt.nn.Linear(dim * 4, dim),
                        pt.nn.GELU(),
                        pt.nn.Dropout(dropout),
                        pt.nn.Linear(dim, dim),
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.edge_norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(int(num_layers))])
            self.ieconv_gates = pt.nn.ModuleList(
                [
                    pt.nn.Sequential(
                        pt.nn.LayerNorm(dim * 3),
                        pt.nn.Linear(dim * 3, dim),
                        pt.nn.GELU(),
                        pt.nn.Dropout(dropout),
                        pt.nn.Linear(dim, dim),
                        pt.nn.Sigmoid(),
                    )
                    for _ in range(int(num_layers))
                ]
            )
        self.local_out = pt.nn.Sequential(pt.nn.LayerNorm(dim + vector_dim), pt.nn.Linear(dim + vector_dim, dim))
        self.last_diagnostics = {}

    def _relation_edge_update(self, s, edge_s, nn_idx, rel_type, layer_idx):
        if self.relation_embedding is None or rel_type is None or edge_s.numel() == 0:
            return edge_s, edge_s.new_tensor(0.0)
        rel_type = rel_type.to(device=edge_s.device, dtype=pt.long).clamp(0, self.num_relations - 1)
        base_s = s.unsqueeze(1).expand(-1, nn_idx.shape[1], -1)
        neigh_s = s[nn_idx]
        rel_emb = self.relation_embedding(rel_type).to(dtype=edge_s.dtype)
        update = self.edge_updates[layer_idx](pt.cat([base_s, neigh_s, edge_s, rel_emb], dim=-1))
        if self.gear_view_mode == "rel_edge_msg_ieconv":
            # IEConv-lite: the relation edge message is modulated by a dynamic
            # per-channel filter generated from the two endpoint states and the
            # current edge state.
            update = update * self.ieconv_gates[layer_idx](pt.cat([base_s, neigh_s, edge_s], dim=-1))
        return self.edge_norms[layer_idx](edge_s + self.dropout(update)), update.detach().norm(dim=-1).mean()

    def forward(self, s0, v0, nn_idx, edge_s, edge_v, rel_type=None, centers=None, frame=None):
        s, v = s0, v0
        update_norms = []
        oagnn_diag_values = {}
        pair_update_norms = []
        pair_gate_values = []
        if nn_idx.shape[1] > 0:
            for layer_idx, layer in enumerate(self.layers):
                edge_s, update_norm = self._relation_edge_update(s, edge_s, nn_idx, rel_type, layer_idx)
                if self.gear_view_mode != "none":
                    update_norms.append(update_norm)
                if self.edge_pair_mode != "none":
                    edge_s, pair_norm, pair_gate = self.pair_updates[layer_idx](edge_s, s, nn_idx)
                    pair_update_norms.append(pair_norm)
                    pair_gate_values.append(pair_gate)
                if self.oagnn_mode == "official":
                    s, v = layer(s, v, edge_s, edge_v, nn_idx, frame=frame)
                    for key, value in getattr(layer, "last_diagnostics", {}).items():
                        oagnn_diag_values.setdefault(key, []).append(value.detach())
                elif self.oagnn_mode == "oa_edge":
                    s, v = layer(s, v, edge_s, edge_v, nn_idx, centers=centers, rel_type=rel_type)
                    for key, value in getattr(layer, "last_diagnostics", {}).items():
                        oagnn_diag_values.setdefault(key, []).append(value.detach())
                else:
                    s, v = layer(s, v, edge_s, edge_v, nn_idx)
        if self.gear_view_mode != "none" and rel_type is not None and rel_type.numel() > 0:
            rel = rel_type.to(device=edge_s.device, dtype=pt.long).clamp(0, self.num_relations - 1)
            rel_counts = pt.nn.functional.one_hot(rel, num_classes=self.num_relations).to(dtype=edge_s.dtype).sum(dim=(0, 1))
            rel_prob = rel_counts / rel_counts.sum().clamp_min(1.0)
            rel_entropy = -(rel_prob * rel_prob.clamp_min(1e-8).log()).sum()
            self.last_diagnostics = {
                "gear_view_edge_update_norm": pt.stack(update_norms).mean() if update_norms else edge_s.new_tensor(0.0),
                "gear_view_relation_entropy": rel_entropy.detach(),
                "gear_view_edge_count_mean": edge_s.new_tensor(float(nn_idx.shape[1])),
                "gear_view_ieconv_enabled": edge_s.new_tensor(1.0 if self.gear_view_mode == "rel_edge_msg_ieconv" else 0.0),
            }
        else:
            self.last_diagnostics = {}
        if oagnn_diag_values:
            self.last_diagnostics.update({key: pt.stack(values).mean() for key, values in oagnn_diag_values.items()})
        if pair_update_norms:
            self.last_diagnostics.update(
                {
                    "edge_pair_update_norm": pt.stack(pair_update_norms).mean(),
                    "edge_pair_gate_mean": pt.stack(pair_gate_values).mean(),
                    "edge_pair_count_mean": edge_s.new_tensor(float(nn_idx.shape[1])),
                }
            )
        return self.local_out(pt.cat([s, pt.linalg.norm(v, dim=-1)], dim=-1))


class CrossViewEdgeMessage(pt.nn.Module):
    def __init__(self, dim, num_views, heads=4, beta=0.1, dropout=0.1):
        super().__init__()
        self.num_views = int(num_views)
        self.beta = float(beta)
        heads = max(1, int(heads))
        if dim % heads != 0:
            heads = 1
        self.view_embedding = pt.nn.Embedding(self.num_views, dim)
        self.attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.norm = pt.nn.LayerNorm(dim)
        self.last_diagnostics = {}

    def forward(self, edge_states):
        if self.num_views <= 1 or not edge_states:
            return edge_states
        names = list(edge_states.keys())
        outputs = {}
        update_norms = []
        gate_means = []
        for target_idx, name in enumerate(names):
            edge_s = edge_states[name]
            if edge_s.numel() == 0:
                outputs[name] = edge_s
                continue
            query = edge_s + self.view_embedding.weight[target_idx].to(device=edge_s.device, dtype=edge_s.dtype).view(1, 1, -1)
            kv_parts = []
            for source_idx, source_name in enumerate(names):
                if source_name == name:
                    continue
                source_edge = edge_states[source_name]
                if source_edge.numel() == 0:
                    continue
                view_bias = self.view_embedding.weight[source_idx].to(device=edge_s.device, dtype=edge_s.dtype).view(1, 1, -1)
                kv_parts.append(source_edge + view_bias)
            if not kv_parts:
                outputs[name] = edge_s
                continue
            key_value = pt.cat(kv_parts, dim=1)
            context, _ = self.attn(query, key_value, key_value, need_weights=False)
            update_input = pt.cat([edge_s, context], dim=-1)
            gate = self.gate(update_input)
            update = self.update(update_input)
            outputs[name] = self.norm(edge_s + self.beta * gate * update)
            update_norms.append(update.detach().norm(dim=-1).mean())
            gate_means.append(gate.detach().mean())
        self.last_diagnostics = {
            "cross_view_edge_update_norm": pt.stack(update_norms).mean() if update_norms else next(iter(edge_states.values())).new_tensor(0.0),
            "cross_view_edge_gate_mean": pt.stack(gate_means).mean() if gate_means else next(iter(edge_states.values())).new_tensor(0.0),
            "cross_view_edge_beta": next(iter(edge_states.values())).new_tensor(self.beta),
        }
        return outputs


class StaticMultiViewGVPEncoder(pt.nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        dim=128,
        vector_dim=16,
        graph_views=("seq", "k16", "k32"),
        view_layers=2,
        rbf_bins=16,
        dropout=0.1,
        extra_residue_feature_dim=0,
        router_mode="global",
        gear_view_mode="none",
        oagnn_mode="none",
        token_encoder_mode="none",
        token_encoder_layers=2,
        token_encoder_heads=4,
        edge_pair_mode="none",
        cross_view_edge_mode="none",
        cross_view_edge_beta=0.1,
        cross_view_edge_heads=4,
        sequence_pos_enc_mode="none",
        sequence_pos_enc_dim=16,
    ):
        super().__init__()
        self.graph_views = tuple(str(x).strip() for x in graph_views if str(x).strip())
        if not self.graph_views:
            raise ValueError("graph_views must contain at least one view.")
        self.router_mode = str(router_mode)
        self.gear_view_mode = str(gear_view_mode)
        self.oagnn_mode = str(oagnn_mode)
        self.edge_pair_mode = str(edge_pair_mode)
        self.cross_view_edge_mode = str(cross_view_edge_mode)
        self.feature_builder = ResidueFeatureBuilder(
            atom_feature_dim,
            dim,
            vector_dim,
            rbf_bins=rbf_bins,
            dropout=dropout,
            extra_residue_feature_dim=extra_residue_feature_dim,
            token_encoder_mode=token_encoder_mode,
            token_encoder_layers=token_encoder_layers,
            token_encoder_heads=token_encoder_heads,
            sequence_pos_enc_mode=sequence_pos_enc_mode,
            sequence_pos_enc_dim=sequence_pos_enc_dim,
        )
        self.views = pt.nn.ModuleDict(
            {
                name: StaticGraphView(
                    dim,
                    vector_dim,
                    num_layers=view_layers,
                    dropout=dropout,
                    gear_view_mode=self.gear_view_mode,
                    oagnn_mode=self.oagnn_mode,
                    edge_pair_mode=self.edge_pair_mode,
                )
                for name in self.graph_views
            }
        )
        if self.cross_view_edge_mode not in {"none", "source_residue_attn"}:
            raise ValueError("cross_view_edge_mode must be one of: none, source_residue_attn")
        self.cross_view_edge_layers = pt.nn.ModuleList()
        if self.cross_view_edge_mode != "none":
            self.cross_view_edge_layers = pt.nn.ModuleList(
                [
                    CrossViewEdgeMessage(
                        dim=dim,
                        num_views=len(self.graph_views),
                        heads=cross_view_edge_heads,
                        beta=cross_view_edge_beta,
                        dropout=dropout,
                    )
                    for _ in range(int(view_layers))
                ]
            )
        if self.router_mode not in {"global", "per_protein", "per_residue", "full_attention"}:
            raise ValueError("router_mode must be one of: global, per_protein, per_residue, full_attention")
        if self.router_mode == "global":
            self.view_logits = pt.nn.Parameter(pt.zeros(len(self.graph_views)))
        elif self.router_mode == "per_protein":
            with pt.random.fork_rng(devices=[]):
                self.protein_router = self._build_per_protein_router(
                    dim, len(self.graph_views), dropout
                )
        elif self.router_mode == "per_residue":
            self.view_router = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * len(self.graph_views)),
                pt.nn.Linear(dim * len(self.graph_views), dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, len(self.graph_views)),
            )
        else:
            attn_heads = 4 if dim % 4 == 0 else 1
            self.residue_query = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
            self.view_attn_rounds = pt.nn.ModuleList(
                [
                    pt.nn.MultiheadAttention(
                        embed_dim=dim,
                        num_heads=attn_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(2)
                ]
            )
            self.residue_updates = pt.nn.ModuleList(
                [
                    pt.nn.Sequential(
                        pt.nn.LayerNorm(dim * 2),
                        pt.nn.Linear(dim * 2, dim),
                        pt.nn.GELU(),
                        pt.nn.Dropout(dropout),
                        pt.nn.Linear(dim, dim),
                    )
                    for _ in range(2)
                ]
            )
            self.attn_round_norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(2)])
        self.fusion_norm = pt.nn.LayerNorm(dim)
        self.router_entropy_eps = 1e-8
        self.last_diagnostics = {}

    @staticmethod
    def _build_per_protein_router(dim, num_views, dropout):
        router = pt.nn.Sequential(
            pt.nn.LayerNorm(int(dim) * int(num_views) * 2),
            pt.nn.Linear(int(dim) * int(num_views) * 2, int(dim)),
            pt.nn.GELU(),
            pt.nn.Dropout(float(dropout)),
            pt.nn.Linear(int(dim), int(num_views)),
        )
        pt.nn.init.zeros_(router[-1].weight)
        pt.nn.init.zeros_(router[-1].bias)
        return router

    def protein_router_weights(self, view_outputs):
        if not view_outputs:
            raise ValueError("per_protein router requires at least one graph view")
        stacked = pt.stack(view_outputs, dim=0)
        if stacked.shape[1] == 0:
            raise ValueError("per_protein router requires at least one residue")
        pooled = pt.cat(
            [stacked.mean(dim=1), stacked.amax(dim=1)], dim=-1
        ).reshape(-1)
        return pt.softmax(self.protein_router(pooled), dim=-1).to(
            dtype=stacked.dtype, device=stacked.device
        )

    def _forward_cross_view(self, s0, v0, view_inputs, centers=None, frame=None):
        states = {
            name: {
                "s": s0,
                "v": v0,
                "edge_s": values["edge_s"],
                "edge_v": values["edge_v"],
                "nn_idx": values["nn_idx"],
                "rel_type": values["rel_type"],
            }
            for name, values in view_inputs.items()
        }
        diag_values = {}
        for layer_idx in range(len(self.cross_view_edge_layers)):
            edge_states = {}
            for name, state in states.items():
                view = self.views[name]
                edge_s, update_norm = view._relation_edge_update(
                    state["s"],
                    state["edge_s"],
                    state["nn_idx"],
                    state["rel_type"],
                    layer_idx,
                )
                state["edge_s"] = edge_s
                if view.gear_view_mode != "none":
                    diag_values.setdefault("gear_view_edge_update_norm", []).append(update_norm.detach())
                if view.edge_pair_mode != "none":
                    edge_s, pair_norm, pair_gate = view.pair_updates[layer_idx](state["edge_s"], state["s"], state["nn_idx"])
                    state["edge_s"] = edge_s
                    diag_values.setdefault("edge_pair_update_norm", []).append(pair_norm.detach())
                    diag_values.setdefault("edge_pair_gate_mean", []).append(pair_gate.detach())
                edge_states[name] = state["edge_s"]
            edge_states = self.cross_view_edge_layers[layer_idx](edge_states)
            for key, value in getattr(self.cross_view_edge_layers[layer_idx], "last_diagnostics", {}).items():
                diag_values.setdefault(key, []).append(value.detach())
            for name, edge_s in edge_states.items():
                states[name]["edge_s"] = edge_s
            for name, state in states.items():
                view = self.views[name]
                layer = view.layers[layer_idx]
                if view.oagnn_mode == "official":
                    state["s"], state["v"] = layer(
                        state["s"],
                        state["v"],
                        state["edge_s"],
                        state["edge_v"],
                        state["nn_idx"],
                        frame=frame,
                    )
                    for key, value in getattr(layer, "last_diagnostics", {}).items():
                        diag_values.setdefault(key, []).append(value.detach())
                elif view.oagnn_mode == "oa_edge":
                    state["s"], state["v"] = layer(
                        state["s"],
                        state["v"],
                        state["edge_s"],
                        state["edge_v"],
                        state["nn_idx"],
                        centers=centers,
                        rel_type=state["rel_type"],
                    )
                    for key, value in getattr(layer, "last_diagnostics", {}).items():
                        diag_values.setdefault(key, []).append(value.detach())
                else:
                    state["s"], state["v"] = layer(state["s"], state["v"], state["edge_s"], state["edge_v"], state["nn_idx"])
        view_outputs = []
        for name in self.graph_views:
            state = states[name]
            view_outputs.append(self.views[name].local_out(pt.cat([state["s"], pt.linalg.norm(state["v"], dim=-1)], dim=-1)))
        self.last_diagnostics = {key: pt.stack(values).mean() for key, values in diag_values.items() if values}
        return view_outputs

    def _view_edges(self, name, centers, graph_cache, true_feature_cache=None, chain_ids=None):
        if name == "seq":
            return self.feature_builder.build_seq_edges(centers, chain_ids=chain_ids)
        if name in {"seq_chain", "chainseq", "chain_seq"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(1,), chain_ids=chain_ids)
        if name in {"seq_short", "seqshort"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(2, 4), chain_ids=chain_ids)
        if name in {"seq_mid", "seqmid"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(8, 16), chain_ids=chain_ids)
        if name in {"seq_long", "seqlong"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(32, 64), chain_ids=chain_ids)
        if name in {"seq_multi", "seq_multirange", "seqdilated", "seq_dilated"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(1, 2, 4, 8, 16, 32, 64), chain_ids=chain_ids)
        if name in {"ss", "ss_graph", "secondary", "secondary_structure"}:
            return self.feature_builder.build_ss_edges(centers, true_feature_cache, k=12)
        if name.startswith("ss"):
            suffix = name[2:]
            if suffix.isdigit():
                return self.feature_builder.build_ss_edges(centers, true_feature_cache, k=int(suffix))
        if name in {"contact", "contact10"}:
            return self.feature_builder.build_contact_edges(centers, threshold=10.0, max_k=32)
        if name == "contact8":
            return self.feature_builder.build_contact_edges(centers, threshold=8.0, max_k=32)
        if name == "contact12":
            return self.feature_builder.build_contact_edges(centers, threshold=12.0, max_k=32)
        if name.startswith("contact"):
            suffix = name.replace("contact", "")
            if suffix.isdigit():
                return self.feature_builder.build_contact_edges(centers, threshold=float(suffix), max_k=32)
        if name in {"exposure", "pseudoexposure", "pexp"}:
            return self.feature_builder.build_exposure_edges(centers, k=16)
        if name.startswith("exposure"):
            return self.feature_builder.build_exposure_edges(centers, k=int(name.replace("exposure", "") or 16))
        if name in {"true_sasa", "true_dssp", "true_feature", "true"}:
            if true_feature_cache is None:
                raise ValueError("true_feature_cache is required for true SASA/DSSP view.")
            feat = true_feature_cache.to(device=centers.device, dtype=centers.dtype)
            if feat.shape[0] != centers.shape[0]:
                raise ValueError(
                    f"true_feature cache length mismatch: {feat.shape[0]} vs {centers.shape[0]}"
                )
            if feat.shape[1] >= 4:
                exposure = feat[:, 3:4]
            else:
                exposure = centers.new_zeros((centers.shape[0], 1))
            ss = feat[:, :3] if feat.shape[1] >= 3 else centers.new_zeros((centers.shape[0], 3))
            exposure_dist = pt.cdist(exposure, exposure).clamp_min(0.0)
            ss_sim = 1.0 - pt.matmul(ss, ss.transpose(0, 1)).clamp(-1.0, 1.0)
            score = exposure_dist + 0.25 * ss_sim
            score.fill_diagonal_(float("inf"))
            kk = min(16, max(centers.shape[0] - 1, 1))
            _, idx = pt.topk(score, k=kk, dim=-1, largest=False)
            dist = pt.gather(score, 1, idx)
            return idx.contiguous(), dist.contiguous()
        if name.startswith("k"):
            return self.feature_builder.build_knn_edges(centers, graph_cache, int(name[1:]))
        raise ValueError(f"Unsupported graph view: {name}")

    def _edge_relation_types(self, name, centers, nn_idx, nn_dist):
        if (self.gear_view_mode == "none" and self.oagnn_mode == "none") or nn_idx.shape[1] == 0:
            return None
        base = pt.arange(centers.shape[0], device=centers.device).unsqueeze(-1)
        sep = (nn_idx - base).abs()
        rel = pt.full(nn_idx.shape, 7, device=centers.device, dtype=pt.long)
        if name == "seq" or name.startswith("seq") or name in {"chainseq", "chain_seq"}:
            rel = pt.where((nn_idx - base).sign() < 0, pt.full_like(rel, 0), pt.full_like(rel, 1))
        elif name in {"exposure", "pseudoexposure", "pexp"} or name.startswith("exposure"):
            rel = pt.full_like(rel, 6)
        elif name.startswith("k") or name.startswith("contact"):
            rel = pt.where(nn_dist < 18.0, pt.full_like(rel, 5), rel)
            rel = pt.where(nn_dist < 12.0, pt.full_like(rel, 4), rel)
            rel = pt.where(nn_dist < 8.0, pt.full_like(rel, 3), rel)
        elif name in {"ss", "ss_graph", "secondary", "secondary_structure"} or name.startswith("ss"):
            rel = pt.full_like(rel, 6)
        else:
            rel = pt.full_like(rel, 7)
        rel = pt.where(sep <= 4, pt.minimum(rel, pt.full_like(rel, 2)), rel)
        return rel.clamp(0, 7)

    def forward(self, X, q_atom, M=None, graph_cache=None, extra_residue_features=None, true_feature_cache=None, chain_ids=None, atom_to_residue_index=None, num_residues=None, static_geometry=None):
        s0, v0, centers = self.feature_builder.node_features(
            X, q_atom, M, extra_residue_features=extra_residue_features,
            atom_to_residue_index=atom_to_residue_index, num_residues=num_residues,
            static_geometry=static_geometry,
        )
        frame = None
        if self.oagnn_mode == "official":
            frame, _pos_ca, frame_valid, _centers_from_ca = self.feature_builder.backbone_frames(
                X, M, q_atom, atom_to_residue_index=atom_to_residue_index, num_residues=num_residues
            )
            frame = frame.to(device=s0.device, dtype=s0.dtype)
        self.last_residue_vectors = v0
        view_outputs = []
        gear_diag_values = {}
        for key, value in getattr(self.feature_builder, "last_diagnostics", {}).items():
            gear_diag_values.setdefault(key, []).append(value.detach())
        if self.oagnn_mode == "official":
            gear_diag_values.setdefault("official_oagnn_frame_valid_mean", []).append(frame_valid.to(device=s0.device, dtype=s0.dtype).mean().detach())
        view_inputs = {}
        for name in self.graph_views:
            nn_idx, nn_dist = self._view_edges(name, centers, graph_cache, true_feature_cache=true_feature_cache, chain_ids=chain_ids)
            edge_s, edge_v = self.feature_builder.edge_features(centers, nn_idx, nn_dist)
            rel_type = self._edge_relation_types(name, centers, nn_idx, nn_dist)
            view_inputs[name] = {
                "nn_idx": nn_idx,
                "edge_s": edge_s,
                "edge_v": edge_v,
                "rel_type": rel_type,
            }
        if self.cross_view_edge_mode == "none":
            for name, values in view_inputs.items():
                view_outputs.append(
                    self.views[name](
                        s0,
                        v0,
                        values["nn_idx"],
                        values["edge_s"],
                        values["edge_v"],
                        rel_type=values["rel_type"],
                        centers=centers,
                        frame=frame,
                    )
                )
                for key, value in getattr(self.views[name], "last_diagnostics", {}).items():
                    gear_diag_values.setdefault(key, []).append(value.detach())
            if gear_diag_values:
                self.last_diagnostics = {key: pt.stack(values).mean() for key, values in gear_diag_values.items()}
            else:
                self.last_diagnostics = {}
        else:
            view_outputs = self._forward_cross_view(s0, v0, view_inputs, centers=centers, frame=frame)
        stacked = pt.stack(view_outputs, dim=0)
        if self.router_mode == "global":
            weights = pt.softmax(self.view_logits, dim=0).to(dtype=stacked.dtype, device=stacked.device)
            fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
            router_entropy = -(weights * (weights.clamp_min(self.router_entropy_eps).log())).sum()
            router_map = weights.unsqueeze(0).expand(stacked.shape[1], -1)
        elif self.router_mode == "per_protein":
            weights = self.protein_router_weights(view_outputs)
            fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
            router_entropy = -(weights * weights.clamp_min(self.router_entropy_eps).log()).sum()
            router_map = weights.unsqueeze(0).expand(stacked.shape[1], -1)
        elif self.router_mode == "per_residue":
            weights = None
            per_residue = pt.cat(view_outputs, dim=-1)
            router_logits = self.view_router(per_residue)
            router_map = pt.softmax(router_logits, dim=-1)
            fused = (router_map.unsqueeze(-1) * stacked.permute(1, 0, 2)).sum(dim=1)
            router_entropy = -(router_map * router_map.clamp_min(self.router_entropy_eps).log()).sum(dim=-1).mean()
        else:
            weights = None
            view_tokens = stacked.permute(1, 0, 2).contiguous()
            query = self.residue_query(s0).unsqueeze(1)
            router_map = None
            fused = None
            for attn, update, norm in zip(self.view_attn_rounds, self.residue_updates, self.attn_round_norms):
                attn_out, attn_weights = attn(query, view_tokens, view_tokens, need_weights=True, average_attn_weights=True)
                query = norm(query + update(pt.cat([query, attn_out], dim=-1)))
                fused = query.squeeze(1)
                router_map = attn_weights.squeeze(1)
            router_entropy = -(router_map * router_map.clamp_min(self.router_entropy_eps).log()).sum(dim=-1).mean()
        return self.fusion_norm(fused).unsqueeze(0), stacked, weights, router_map, router_entropy


class StaticResidueMultiplexGVPEncoder(pt.nn.Module):
    """Layer-wise residue multiplex graph encoder.

    The residue nodes share one scalar/vector state while each graph view keeps
    its own edge set and message parameters. At every layer, view-specific
    updates are gated per residue and fused back into the shared residue state.
    """

    def __init__(
        self,
        atom_feature_dim,
        dim=128,
        vector_dim=16,
        graph_views=("seq", "k16", "k32"),
        view_layers=2,
        rbf_bins=16,
        dropout=0.1,
        extra_residue_feature_dim=0,
        router_mode="per_residue",
        gear_view_mode="none",
        oagnn_mode="none",
        token_encoder_mode="none",
        token_encoder_layers=2,
        token_encoder_heads=4,
        edge_pair_mode="none",
        sequence_pos_enc_mode="none",
        sequence_pos_enc_dim=16,
    ):
        super().__init__()
        self.graph_views = tuple(str(x).strip() for x in graph_views if str(x).strip())
        if not self.graph_views:
            raise ValueError("graph_views must contain at least one view.")
        self.router_mode = str(router_mode)
        if self.router_mode not in {"global", "per_residue"}:
            raise ValueError("multiplex router_mode must be one of: global, per_residue")
        self.gear_view_mode = str(gear_view_mode)
        self.oagnn_mode = str(oagnn_mode)
        self.edge_pair_mode = str(edge_pair_mode)
        if self.oagnn_mode != "none" or self.gear_view_mode != "none" or self.edge_pair_mode != "none":
            raise ValueError("StaticResidueMultiplexGVPEncoder currently expects oagnn/gear_view/edge_pair modes to be none.")
        self.feature_builder = ResidueFeatureBuilder(
            atom_feature_dim,
            dim,
            vector_dim,
            rbf_bins=rbf_bins,
            dropout=dropout,
            extra_residue_feature_dim=extra_residue_feature_dim,
            token_encoder_mode=token_encoder_mode,
            token_encoder_layers=token_encoder_layers,
            token_encoder_heads=token_encoder_heads,
            sequence_pos_enc_mode=sequence_pos_enc_mode,
            sequence_pos_enc_dim=sequence_pos_enc_dim,
        )
        self.view_layers = pt.nn.ModuleDict(
            {
                name: pt.nn.ModuleList(
                    [FullGVPMessageLayer((dim, vector_dim), (dim, vector_dim), dropout=dropout) for _ in range(int(view_layers))]
                )
                for name in self.graph_views
            }
        )
        self.scalar_projs = pt.nn.ModuleDict(
            {
                name: pt.nn.ModuleList(
                    [
                        pt.nn.Sequential(
                            pt.nn.LayerNorm(dim),
                            pt.nn.Linear(dim, dim),
                            pt.nn.GELU(),
                            pt.nn.Dropout(dropout),
                            pt.nn.Linear(dim, dim),
                        )
                        for _ in range(int(view_layers))
                    ]
                )
                for name in self.graph_views
            }
        )
        if self.router_mode == "global":
            self.view_logits = pt.nn.ParameterList([pt.nn.Parameter(pt.zeros(len(self.graph_views))) for _ in range(int(view_layers))])
        else:
            self.view_gates = pt.nn.ModuleList(
                [
                    pt.nn.Sequential(
                        pt.nn.LayerNorm(dim * (len(self.graph_views) + 1)),
                        pt.nn.Linear(dim * (len(self.graph_views) + 1), dim),
                        pt.nn.GELU(),
                        pt.nn.Dropout(dropout),
                        pt.nn.Linear(dim, len(self.graph_views)),
                    )
                    for _ in range(int(view_layers))
                ]
            )
        self.scalar_norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(int(view_layers))])
        self.dropout = pt.nn.Dropout(dropout)
        self.local_out = pt.nn.Sequential(pt.nn.LayerNorm(dim + vector_dim), pt.nn.Linear(dim + vector_dim, dim))
        self.fusion_norm = pt.nn.LayerNorm(dim)
        self.router_entropy_eps = 1e-8
        self.last_diagnostics = {}
        self.last_residue_vectors = None

    def _view_edges(self, name, centers, graph_cache, true_feature_cache=None, chain_ids=None):
        if name == "seq":
            return self.feature_builder.build_seq_edges(centers)
        if name in {"seq_chain", "chainseq", "chain_seq"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(1,), chain_ids=chain_ids)
        if name in {"seq_short", "seqshort"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(2, 4), chain_ids=chain_ids)
        if name in {"seq_mid", "seqmid"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(8, 16), chain_ids=chain_ids)
        if name in {"seq_long", "seqlong"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(32, 64), chain_ids=chain_ids)
        if name in {"seq_multi", "seq_multirange", "seqdilated", "seq_dilated"}:
            return self.feature_builder.build_seq_edges(centers, offsets=(1, 2, 4, 8, 16, 32, 64), chain_ids=chain_ids)
        if name in {"ss", "ss_graph", "secondary", "secondary_structure"}:
            return self.feature_builder.build_ss_edges(centers, true_feature_cache, k=12)
        if name.startswith("ss"):
            suffix = name[2:]
            if suffix.isdigit():
                return self.feature_builder.build_ss_edges(centers, true_feature_cache, k=int(suffix))
        if name in {"contact", "contact10"}:
            return self.feature_builder.build_contact_edges(centers, threshold=10.0, max_k=32)
        if name == "contact8":
            return self.feature_builder.build_contact_edges(centers, threshold=8.0, max_k=32)
        if name == "contact12":
            return self.feature_builder.build_contact_edges(centers, threshold=12.0, max_k=32)
        if name.startswith("contact"):
            suffix = name.replace("contact", "")
            if suffix.isdigit():
                return self.feature_builder.build_contact_edges(centers, threshold=float(suffix), max_k=32)
        if name in {"exposure", "pseudoexposure", "pexp"}:
            return self.feature_builder.build_exposure_edges(centers, k=16)
        if name.startswith("exposure"):
            return self.feature_builder.build_exposure_edges(centers, k=int(name.replace("exposure", "") or 16))
        if name in {"true_sasa", "true_dssp", "true_feature", "true"}:
            if true_feature_cache is None:
                raise ValueError("true_feature_cache is required for true SASA/DSSP view.")
            feat = true_feature_cache.to(device=centers.device, dtype=centers.dtype)
            exposure = feat[:, 3:4] if feat.shape[1] >= 4 else centers.new_zeros((centers.shape[0], 1))
            ss = feat[:, :3] if feat.shape[1] >= 3 else centers.new_zeros((centers.shape[0], 3))
            score = pt.cdist(exposure, exposure).clamp_min(0.0) + 0.25 * (1.0 - pt.matmul(ss, ss.transpose(0, 1)).clamp(-1.0, 1.0))
            score.fill_diagonal_(float("inf"))
            kk = min(16, max(centers.shape[0] - 1, 1))
            dist, idx = pt.topk(score, k=kk, dim=-1, largest=False)
            return idx.contiguous(), dist.contiguous()
        if name.startswith("k"):
            return self.feature_builder.build_knn_edges(centers, graph_cache, int(name[1:]))
        raise ValueError(f"Unsupported graph view: {name}")

    def forward(self, X, q_atom, M=None, graph_cache=None, extra_residue_features=None, true_feature_cache=None, chain_ids=None, atom_to_residue_index=None, num_residues=None, static_geometry=None):
        s, v, centers = self.feature_builder.node_features(
            X, q_atom, M, extra_residue_features=extra_residue_features,
            atom_to_residue_index=atom_to_residue_index, num_residues=num_residues,
            static_geometry=static_geometry,
        )
        view_data = {}
        for name in self.graph_views:
            nn_idx, nn_dist = self._view_edges(name, centers, graph_cache, true_feature_cache=true_feature_cache, chain_ids=chain_ids)
            edge_s, edge_v = self.feature_builder.edge_features(centers, nn_idx, nn_dist)
            view_data[name] = (nn_idx, edge_s, edge_v)
        router_maps = []
        router_entropies = []
        update_norms = []
        for layer_idx in range(len(self.scalar_norms)):
            msg_s_list = []
            msg_v_list = []
            for name in self.graph_views:
                nn_idx, edge_s, edge_v = view_data[name]
                s_next, v_next = self.view_layers[name][layer_idx](s, v, edge_s, edge_v, nn_idx)
                msg_s = self.scalar_projs[name][layer_idx](s_next - s)
                msg_v = v_next - v
                msg_s_list.append(msg_s)
                msg_v_list.append(msg_v)
            stacked_s = pt.stack(msg_s_list, dim=1)
            stacked_v = pt.stack(msg_v_list, dim=1)
            if self.router_mode == "global":
                alpha = pt.softmax(self.view_logits[layer_idx], dim=0).to(device=s.device, dtype=s.dtype)
                router_map = alpha.unsqueeze(0).expand(s.shape[0], -1)
            else:
                gate_input = pt.cat([s, *msg_s_list], dim=-1)
                router_map = pt.softmax(self.view_gates[layer_idx](gate_input), dim=-1)
            update_s = (router_map.unsqueeze(-1) * stacked_s).sum(dim=1)
            update_v = (router_map.unsqueeze(-1).unsqueeze(-1) * stacked_v).sum(dim=1)
            s = self.scalar_norms[layer_idx](s + self.dropout(update_s))
            v = v + update_v
            router_maps.append(router_map)
            router_entropies.append(-(router_map * router_map.clamp_min(self.router_entropy_eps).log()).sum(dim=-1).mean())
            update_norms.append(update_s.detach().norm(dim=-1).mean())
        h = self.fusion_norm(self.local_out(pt.cat([s, pt.linalg.norm(v, dim=-1)], dim=-1)))
        self.last_residue_vectors = v
        router_map = pt.stack(router_maps).mean(dim=0) if router_maps else None
        router_entropy = pt.stack(router_entropies).mean() if router_entropies else h.new_tensor(0.0)
        self.last_diagnostics = {
            "multiplex_update_norm": pt.stack(update_norms).mean() if update_norms else h.new_tensor(0.0),
            "multiplex_router_entropy": router_entropy.detach(),
        }
        return h.unsqueeze(0), pt.stack(msg_s_list, dim=0), None, router_map, router_entropy


class GlobalContextFeedback(pt.nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h, mask):
        mask = mask.to(dtype=h.dtype, device=h.device)
        mean = masked_mean(h, mask, dim=1)
        masked_h = h.masked_fill(mask.unsqueeze(-1) < 0.5, pt.finfo(h.dtype).min)
        maxv = masked_h.max(dim=1).values
        maxv = pt.where(pt.isfinite(maxv), maxv, pt.zeros_like(maxv))
        g = 0.5 * (mean + maxv)
        g_expand = g.unsqueeze(1).expand_as(h)
        x = pt.cat([h, g_expand, h * g_expand], dim=-1)
        return self.norm(h + self.gate(x) * self.update(x))


class InterfaceHead(pt.nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.net = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)
