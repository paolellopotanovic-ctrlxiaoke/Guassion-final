"""Surface graph encoders and patch operators used by M4."""

import math

import torch as pt

from .gvp import FullGVPMessageLayer
from .runtime import replay_dropout_rng


class SurfacePointGraphEncoder(pt.nn.Module):
    def __init__(self, dim=128, geom_dim=6, dropout=0.1, layers=2):
        super().__init__()
        edge_dim = 8
        self.input_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.edge_mlp = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2 + edge_dim),
                    pt.nn.Linear(dim * 2 + edge_dim, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(int(layers))
            ]
        )
        self.gates = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.Sigmoid(),
                )
                for _ in range(int(layers))
            ]
        )
        self.norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(int(layers))])

    def _edge_features(self, surface_pos, surface_normal, graph_idx, graph_dist):
        p = int(surface_pos.shape[0])
        if graph_idx is None or graph_idx.numel() == 0:
            return surface_pos.new_empty((p, 0, 8))
        idx = graph_idx.to(device=surface_pos.device, dtype=pt.long).clamp(0, max(p - 1, 0))
        neigh_pos = surface_pos[idx]
        rel = neigh_pos - surface_pos.unsqueeze(1)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist
        neigh_normal = surface_normal[idx]
        center_normal = surface_normal.unsqueeze(1).expand_as(neigh_normal)
        normal_dot = (center_normal * neigh_normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        tangent = (direction * center_normal).sum(dim=-1, keepdim=True)
        if graph_dist is None or graph_dist.numel() == 0:
            graph_dist = dist.squeeze(-1)
        graph_dist = graph_dist.to(device=surface_pos.device, dtype=surface_pos.dtype).unsqueeze(-1)
        return pt.cat([dist, graph_dist, direction, normal_dot, tangent, tangent.abs()], dim=-1)

    def forward(self, h_surf0, surface_geom, surface_pos, surface_normal, graph_idx, graph_dist=None):
        geom = surface_geom.to(device=h_surf0.device, dtype=h_surf0.dtype)
        x = self.input_proj(pt.cat([h_surf0, geom], dim=-1))
        surface_pos = surface_pos.to(device=x.device, dtype=x.dtype)
        surface_normal = surface_normal.to(device=x.device, dtype=x.dtype)
        edge_feat = self._edge_features(surface_pos, surface_normal, graph_idx, graph_dist)
        if graph_idx is None or graph_idx.numel() == 0:
            return x
        idx = graph_idx.to(device=x.device, dtype=pt.long).clamp(0, max(x.shape[0] - 1, 0))
        for layer, gate, norm in zip(self.edge_mlp, self.gates, self.norms):
            neigh = x[idx]
            center = x.unsqueeze(1).expand_as(neigh)
            msg = layer(pt.cat([center, neigh, edge_feat.to(device=x.device, dtype=x.dtype)], dim=-1))
            ctx = msg.mean(dim=1)
            x = norm(x + gate(pt.cat([x, ctx], dim=-1)) * ctx)
        return x


class SurfacePointGVPGraphEncoder(pt.nn.Module):
    """Surface-point graph update with scalar/vector GVP message passing."""

    def __init__(self, dim=128, vector_dim=16, geom_dim=6, dropout=0.1, layers=2):
        super().__init__()
        edge_dim = 8
        self.dim = int(dim)
        self.vector_dim = int(vector_dim)
        self.input_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.node_vector_proj = pt.nn.Linear(1, self.vector_dim, bias=False)
        self.edge_scalar_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(edge_dim),
            pt.nn.Linear(edge_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.edge_vector_proj = pt.nn.Linear(1, self.vector_dim, bias=False)
        self.layers = pt.nn.ModuleList(
            [FullGVPMessageLayer((dim, self.vector_dim), (dim, self.vector_dim), dropout=dropout) for _ in range(int(layers))]
        )
        self.last_surface_vectors = None

    def _edge_features(self, surface_pos, surface_normal, graph_idx, graph_dist):
        p = int(surface_pos.shape[0])
        if graph_idx is None or graph_idx.numel() == 0:
            return surface_pos.new_empty((p, 0, 8)), surface_pos.new_empty((p, 0, 1, 3))
        idx = graph_idx.to(device=surface_pos.device, dtype=pt.long).clamp(0, max(p - 1, 0))
        neigh_pos = surface_pos[idx]
        rel = neigh_pos - surface_pos.unsqueeze(1)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist
        neigh_normal = surface_normal[idx]
        center_normal = surface_normal.unsqueeze(1).expand_as(neigh_normal)
        normal_dot = (center_normal * neigh_normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        tangent = (direction * center_normal).sum(dim=-1, keepdim=True)
        if graph_dist is None or graph_dist.numel() == 0:
            graph_dist = dist.squeeze(-1)
        graph_dist = graph_dist.to(device=surface_pos.device, dtype=surface_pos.dtype).unsqueeze(-1)
        edge_s = pt.cat([dist, graph_dist, direction, normal_dot, tangent, tangent.abs()], dim=-1)
        return edge_s, direction.unsqueeze(-2)

    def _project_vectors(self, vectors, proj):
        return proj(vectors.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, h_surf0, surface_geom, surface_pos, surface_normal, graph_idx, graph_dist=None, initial_vectors=None):
        geom = surface_geom.to(device=h_surf0.device, dtype=h_surf0.dtype)
        s = self.input_proj(pt.cat([h_surf0, geom], dim=-1))
        surface_pos = surface_pos.to(device=s.device, dtype=s.dtype)
        surface_normal = surface_normal.to(device=s.device, dtype=s.dtype)
        if initial_vectors is not None and initial_vectors.shape == (s.shape[0], self.vector_dim, 3):
            v = initial_vectors.to(device=s.device, dtype=s.dtype)
        else:
            v = self._project_vectors(surface_normal.unsqueeze(-2), self.node_vector_proj)
        if graph_idx is None or graph_idx.numel() == 0:
            self.last_surface_vectors = v
            return s
        idx = graph_idx.to(device=s.device, dtype=pt.long).clamp(0, max(s.shape[0] - 1, 0))
        edge_feat, edge_vec = self._edge_features(surface_pos, surface_normal, idx, graph_dist)
        edge_s = self.edge_scalar_proj(edge_feat.to(device=s.device, dtype=s.dtype))
        edge_v = self._project_vectors(edge_vec.to(device=s.device, dtype=s.dtype), self.edge_vector_proj)
        for layer in self.layers:
            s, v = layer(s, v, edge_s, edge_v, idx)
        self.last_surface_vectors = v
        return s


class ResidueVectorToSurfaceCrossAttention(pt.nn.Module):
    """Equivariant residue-vector value readout for surface GVP initialization."""

    def __init__(self, dim=128, vector_dim=16, geom_dim=6, residue_topk=16, dropout=0.1):
        super().__init__()
        self.residue_topk = max(1, int(residue_topk))
        self.vector_dim = int(vector_dim)
        score_dim = dim * 2 + geom_dim + 4
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(score_dim),
            pt.nn.Linear(score_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.value_weight = pt.nn.Parameter(pt.eye(self.vector_dim))
        self.last_diagnostics = {}

    def forward(self, h_surf, h_res, residue_vectors, surface_pos, surface_normal, surface_geom, residue_centers):
        self.last_diagnostics = {}
        if residue_vectors is None or residue_vectors.ndim != 3 or residue_vectors.shape[0] != h_res.shape[0]:
            return None
        if residue_vectors.shape[1] != self.vector_dim or residue_vectors.shape[2] != 3:
            return None
        if h_surf.numel() == 0 or h_res.numel() == 0:
            return None

        device = h_surf.device
        dtype = h_surf.dtype
        centers = residue_centers.to(device=device, dtype=dtype)
        surf_pos = surface_pos.to(device=device, dtype=dtype)
        surf_normal = surface_normal.to(device=device, dtype=dtype)
        surf_geom = surface_geom.to(device=device, dtype=dtype)
        res_v_all = residue_vectors.to(device=device, dtype=dtype)

        dist = pt.cdist(surf_pos, centers).clamp_min(0.0)
        k = min(self.residue_topk, centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        local_dist = dist.gather(1, idx).clamp_min(1e-6)
        rel = centers[idx] - surf_pos.unsqueeze(1)
        direction = rel / local_dist.unsqueeze(-1)
        normal = surf_normal / surf_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal_proj = (direction * normal.unsqueeze(1)).sum(dim=-1, keepdim=True)
        tangent_dist = (local_dist.square() - (local_dist * normal_proj.squeeze(-1)).square()).clamp_min(0.0).sqrt()

        surf_expand = h_surf.unsqueeze(1).expand(-1, k, -1)
        res = h_res[idx]
        geom_pair = pt.cat(
            [
                local_dist.unsqueeze(-1),
                tangent_dist.unsqueeze(-1),
                (local_dist * normal_proj.squeeze(-1)).abs().unsqueeze(-1),
                normal_proj,
                surf_geom.unsqueeze(1).expand(-1, k, -1),
            ],
            dim=-1,
        )
        logits = self.score(pt.cat([surf_expand, res, geom_pair], dim=-1)).squeeze(-1)
        attn = pt.softmax(logits, dim=-1)
        res_v = res_v_all[idx]
        mixed_v = pt.einsum("oi,pkic->pkoc", self.value_weight.to(device=device, dtype=dtype), res_v)
        out = (mixed_v * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        entropy = -(attn * attn.clamp_min(1e-9).log()).sum(dim=-1).mean()
        self.last_diagnostics = {
            "residue_vector_surface_attn_entropy": entropy.detach(),
            "residue_vector_surface_attn_max": attn.max(dim=-1).values.mean().detach(),
            "residue_vector_surface_value_norm": out.norm(dim=-1).mean().detach(),
            "residue_vector_surface_topk": h_surf.new_tensor(float(k)).detach(),
        }
        return out


class DynamicSurfacePatchCluster(pt.nn.Module):
    """Compress dense surface points into learnable patch tokens."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        num_patches=32,
        mode="none",
        beta=0.1,
        dropout=0.1,
        attention_heads=4,
        lappe_dim=0,
    ):
        super().__init__()
        self.num_patches = int(num_patches)
        self.mode = str(mode)
        self.beta = float(beta)
        self.lappe_dim = max(0, int(lappe_dim))
        self.assignment = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2 + geom_dim + 3 + self.lappe_dim),
            pt.nn.Linear(dim * 2 + geom_dim + 3 + self.lappe_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.num_patches),
        )
        self.patch_token = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim + 7 + self.lappe_dim),
            pt.nn.Linear(dim + geom_dim + 7 + self.lappe_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.patch_importance = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
            pt.nn.Sigmoid(),
        )
        heads = max(1, int(attention_heads))
        if dim % heads != 0:
            heads = 1
        self.patch_self_attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.residue_patch_attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.patch_norm = pt.nn.LayerNorm(dim)
        self.residue_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        if self.connected_mode:
            for parameter in self.assignment.parameters():
                parameter.requires_grad_(False)

    @property
    def connected_mode(self):
        return self.mode in {"connected_pool", "connected_pool_adj"}

    def enabled(self):
        return self.mode in {
            "gate", "attn", "self_attn", "replace_attn", "independent_attn",
            "connected_pool", "connected_pool_adj",
        } and self.num_patches > 0 and self.beta != 0.0

    @staticmethod
    def _undirected_edges(graph_idx, num_points):
        graph_idx = graph_idx.to(dtype=pt.long)
        if graph_idx.ndim != 2 or graph_idx.shape[0] != num_points:
            raise ValueError(
                f"surface graph must have shape [P, K], got {tuple(graph_idx.shape)} for P={num_points}"
            )
        rows = pt.arange(num_points, device=graph_idx.device, dtype=pt.long).unsqueeze(1).expand_as(graph_idx)
        valid = (graph_idx >= 0) & (graph_idx < num_points) & (graph_idx != rows)
        source = pt.cat([rows[valid], graph_idx[valid]], dim=0)
        target = pt.cat([graph_idx[valid], rows[valid]], dim=0)
        if source.numel() == 0:
            raise ValueError("surface graph has no usable edges")
        edge_code = source * num_points + target
        edge_code = pt.unique(edge_code, sorted=True)
        return edge_code.div(num_points, rounding_mode="floor"), edge_code.remainder(num_points)

    def _euclidean_fps_seeds(self, surface_pos):
        num_points = int(surface_pos.shape[0])
        patch_count = min(self.num_patches, num_points)
        seeds = []
        position = surface_pos.detach().float()
        centered_pos = position - position.mean(dim=0, keepdim=True)
        next_seed = int(pt.argmax(centered_pos.square().sum(dim=-1)).item())
        min_distance_sq = position.new_full((num_points,), float("inf"))
        for _ in range(patch_count):
            seeds.append(next_seed)
            distance_sq = (position - position[next_seed]).square().sum(dim=-1)
            min_distance_sq = pt.minimum(min_distance_sq, distance_sq)
            min_distance_sq[position.new_tensor(seeds, dtype=pt.long)] = -1.0
            if len(seeds) < patch_count:
                next_seed = int(pt.argmax(min_distance_sq).item())
        return surface_pos.new_tensor(seeds, dtype=pt.long)

    def connected_partition(self, surface_pos, surface_graph_idx):
        num_points = int(surface_pos.shape[0])
        if num_points <= 0:
            raise ValueError("connected patch mode requires at least one surface point")
        patch_count = min(self.num_patches, num_points)
        graph_idx = surface_graph_idx.to(device=surface_pos.device, dtype=pt.long)
        with pt.no_grad():
            if num_points == 1:
                source = graph_idx.new_zeros((1,))
                target = graph_idx.new_zeros((1,))
            else:
                source, target = self._undirected_edges(graph_idx, num_points)
            seeds = self._euclidean_fps_seeds(surface_pos)
            labels = source.new_full((num_points,), -1)
            labels[seeds] = pt.arange(patch_count, device=source.device, dtype=pt.long)
            hop_distance = source.new_full((num_points,), -1)
            hop_distance[seeds] = 0
            for hop in range(1, num_points + 1):
                candidate = labels[target]
                candidate = pt.where(candidate >= 0, candidate, candidate.new_full(candidate.shape, patch_count))
                best = labels.new_full((num_points,), patch_count)
                best.scatter_reduce_(0, source, candidate, reduce="amin", include_self=True)
                frontier = (labels < 0) & (best < patch_count)
                if not pt.any(frontier):
                    break
                labels[frontier] = best[frontier]
                hop_distance[frontier] = hop
            if pt.any(labels < 0):
                missing = int((labels < 0).sum().item())
                raise ValueError(f"connected patch propagation left {missing} surface points unassigned")
            assign = pt.nn.functional.one_hot(labels, num_classes=patch_count).to(dtype=surface_pos.dtype)
            patch_adjacency = pt.zeros(
                (patch_count, patch_count), device=surface_pos.device, dtype=pt.bool
            )
            patch_adjacency[labels[source], labels[target]] = True
            patch_adjacency.fill_diagonal_(True)
            patch_sizes = pt.bincount(labels, minlength=patch_count).to(dtype=pt.float32)
            patch_mean = patch_sizes.mean().clamp_min(1.0)
            cross_edge = labels[source] != labels[target]
            patch_degree = patch_adjacency.sum(dim=-1).to(dtype=pt.float32) - 1.0
            diagnostics = {
                "connected_patch_components_mean": surface_pos.new_tensor(1.0),
                "connected_patch_components_max": surface_pos.new_tensor(1.0),
                "connected_patch_size_min": patch_sizes.min().to(device=surface_pos.device),
                "connected_patch_size_max": patch_sizes.max().to(device=surface_pos.device),
                "connected_patch_size_cv": (patch_sizes.std(unbiased=False) / patch_mean).to(device=surface_pos.device),
                "connected_patch_boundary_fraction": cross_edge.float().mean().to(device=surface_pos.device),
                "connected_patch_degree_mean": patch_degree.mean().to(device=surface_pos.device),
                "connected_patch_hop_radius_mean": hop_distance.float().mean().to(device=surface_pos.device),
                "connected_patch_hop_radius_max": hop_distance.max().float().to(device=surface_pos.device),
            }
        return {
            "assign": assign,
            "labels": labels,
            "adjacency": patch_adjacency,
            "hop_distance": hop_distance,
            "surface_graph_idx": graph_idx.detach(),
            "seeds": seeds.detach(),
            "diagnostics": diagnostics,
        }

    def _summary(self, assign, usage, patch_pos, patch_normal, patch_tokens, centered_pos, surface_normal, patch_attn=None, patch_update=None, residue_gate=None, extra_diagnostics=None):
        effective_patch_count = int(assign.shape[1])
        entropy = -(assign.clamp_min(1e-8) * assign.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(max(effective_patch_count, 2)), 1e-8)
        usage_norm = usage / usage.sum().clamp_min(1e-8)
        active = (usage_norm > (0.25 / max(effective_patch_count, 1))).float().sum()
        usage_entropy = -(usage_norm * usage_norm.clamp_min(1e-8).log()).sum()
        usage_entropy = usage_entropy / max(math.log(max(effective_patch_count, 2)), 1e-8)
        displacement = centered_pos.unsqueeze(1) - patch_pos.unsqueeze(0)
        compactness = (assign * displacement.square().sum(dim=-1)).sum() / usage.sum().clamp_min(1e-8)
        normal_alignment = (surface_normal.unsqueeze(1) * patch_normal.unsqueeze(0)).sum(dim=-1).abs()
        normal_consistency = (assign * normal_alignment).sum() / usage.sum().clamp_min(1e-8)
        token_unit = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        token_cos = token_unit.matmul(token_unit.transpose(0, 1))
        offdiag = ~pt.eye(token_cos.shape[0], device=token_cos.device, dtype=pt.bool)
        token_diversity = 1.0 - token_cos[offdiag].mean() if pt.any(offdiag) else token_cos.new_zeros(())
        out = {
            "patch_assignment_entropy": entropy.detach(),
            "patch_count_effective": assign.new_tensor(float(effective_patch_count)).detach(),
            "patch_usage": usage_norm.detach(),
            "patch_usage_active": active.detach(),
            "patch_usage_entropy": usage_entropy.detach(),
            "patch_spatial_compactness": compactness.detach(),
            "patch_normal_consistency": normal_consistency.detach(),
            "patch_token_norm_mean": patch_tokens.norm(dim=-1).mean().detach(),
            "patch_token_diversity": token_diversity.detach(),
        }
        if patch_attn is not None:
            attn = patch_attn.squeeze(0).clamp_min(1e-8)
            attn_entropy = -(attn * attn.log()).sum(dim=-1)
            out.update({
                "patch_residue_attention_entropy": attn_entropy.mean().detach(),
                "patch_residue_attention_max": attn.max(dim=-1).values.mean().detach(),
                "patch_residue_effective_count": attn_entropy.exp().mean().detach(),
            })
        if patch_update is not None:
            out["patch_context_update_norm"] = patch_update.norm(dim=-1).mean().detach()
        if residue_gate is not None:
            out["patch_residue_gate_mean"] = residue_gate.mean().detach()
            out["patch_residue_gate_std"] = residue_gate.std(unbiased=False).detach()
        if extra_diagnostics:
            out.update({key: value.detach() for key, value in extra_diagnostics.items()})
        return out

    def forward_batched(self, requests):
        """Batch fixed-size connected patches while padding only residues."""

        can_batch = self.connected_mode and all(
            request["connected_partition"] is not None for request in requests
        )
        surface_counts = {int(request["h_surf"].shape[0]) for request in requests}
        patch_counts = {
            int(request["connected_partition"]["assign"].shape[1])
            for request in requests
            if request["connected_partition"] is not None
        }
        if not can_batch or len(surface_counts) != 1 or len(patch_counts) != 1:
            return [self.forward(**request) for request in requests]

        device = requests[0]["h_surf"].device
        dtype = requests[0]["h_surf"].dtype
        residue_counts = [int(request["h_res"].shape[0]) for request in requests]
        max_residues = max(residue_counts)
        h_res = pt.nn.utils.rnn.pad_sequence(
            [request["h_res"] for request in requests], batch_first=True
        )
        surf_context = pt.nn.utils.rnn.pad_sequence(
            [request["surf_context"] for request in requests], batch_first=True
        )
        h_surf = pt.stack([request["h_surf"] for request in requests])
        surface_pos = pt.stack(
            [request["surface_pos"].to(device=device, dtype=dtype) for request in requests]
        )
        surface_normal = pt.stack(
            [request["surface_normal"].to(device=device, dtype=dtype) for request in requests]
        )
        surface_geom = pt.stack(
            [request["surface_geom"].to(device=device, dtype=dtype) for request in requests]
        )
        rs_idx = pt.stack(
            [request["rs_idx"].to(device=device, dtype=pt.long) for request in requests]
        )
        assign = pt.stack(
            [
                request["connected_partition"]["assign"].to(
                    device=device, dtype=dtype
                )
                for request in requests
            ]
        )
        usage = assign.sum(dim=1).clamp_min(1e-6)
        assign_t = assign.transpose(1, 2)
        patch_h = pt.bmm(assign_t, h_surf) / usage.unsqueeze(-1)
        centered_pos = surface_pos - surface_pos.mean(dim=1, keepdim=True)
        patch_pos = pt.bmm(assign_t, centered_pos) / usage.unsqueeze(-1)
        patch_normal = pt.bmm(assign_t, surface_normal) / usage.unsqueeze(-1)
        patch_normal = patch_normal / patch_normal.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        patch_geom = pt.bmm(assign_t, surface_geom) / usage.unsqueeze(-1)

        patch_lappe = None
        if self.lappe_dim > 0:
            lappe_values = []
            for request in requests:
                value = request["surface_lappe"]
                if value is None:
                    value = request["h_surf"].new_zeros(
                        (request["h_surf"].shape[0], self.lappe_dim)
                    )
                else:
                    value = value.to(device=device, dtype=dtype)
                    if value.shape[-1] < self.lappe_dim:
                        value = pt.cat(
                            [
                                value,
                                value.new_zeros(
                                    (value.shape[0], self.lappe_dim - value.shape[-1])
                                ),
                            ],
                            dim=-1,
                        )
                    elif value.shape[-1] > self.lappe_dim:
                        value = value[:, : self.lappe_dim]
                lappe_values.append(value)
            lappe = pt.stack(lappe_values)
            patch_lappe = pt.bmm(assign_t, lappe) / usage.unsqueeze(-1)

        density = (usage / max(float(assign.shape[1]), 1.0)).unsqueeze(-1)
        token_parts = [patch_h, patch_pos, patch_normal, patch_geom, density]
        if patch_lappe is not None:
            token_parts.append(patch_lappe)
        patch_tokens = self.patch_token(pt.cat(token_parts, dim=-1))
        patch_states = [request["patch_state"] for request in requests]
        if all(state is not None for state in patch_states):
            patch_tokens = self.patch_norm(
                patch_tokens
                + pt.stack(
                    [state.to(device=device, dtype=dtype) for state in patch_states]
                )
            )

        adjacency = pt.stack(
            [
                request["connected_partition"]["adjacency"].to(
                    device=device, dtype=pt.bool
                )
                for request in requests
            ]
        )
        attention_mask = (~adjacency).repeat_interleave(
            self.patch_self_attn.num_heads, dim=0
        )
        attn_out, _ = self.patch_self_attn(
            patch_tokens,
            patch_tokens,
            patch_tokens,
            attn_mask=attention_mask,
            need_weights=False,
        )
        patch_tokens = self.patch_norm(patch_tokens + attn_out)
        if self.patch_importance is None:
            replay_dropout_rng(
                patch_tokens,
                (*patch_tokens.shape[:-1], int(patch_tokens.shape[-1]) // 2),
                probability=float(self.canonical_replay_dropout_probability),
                training=self.training,
            )
            replay_patch_gate = None
            replay_surface_gate = None
        else:
            replay_patch_gate = self.patch_importance(patch_tokens)
            replay_surface_gate = pt.bmm(assign, replay_patch_gate).squeeze(-1)

        ctx, _ = self.residue_patch_attn(
            h_res, patch_tokens, patch_tokens, need_weights=False
        )
        gate = self.residue_gate(pt.cat([h_res, ctx], dim=-1))
        patch_context = self.beta * gate * ctx
        surf_context_out = surf_context + patch_context

        patch_count = int(assign.shape[-1])
        entropy = -(assign.clamp_min(1e-8) * assign.clamp_min(1e-8).log()).sum(
            dim=-1
        ) / max(math.log(max(patch_count, 2)), 1e-8)
        usage_norm = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        active = (usage_norm > (0.25 / max(patch_count, 1))).float().sum(dim=-1)
        usage_entropy = -(
            usage_norm * usage_norm.clamp_min(1e-8).log()
        ).sum(dim=-1) / max(math.log(max(patch_count, 2)), 1e-8)
        displacement = centered_pos.unsqueeze(2) - patch_pos.unsqueeze(1)
        compactness = (
            assign * displacement.square().sum(dim=-1)
        ).sum(dim=(1, 2)) / usage.sum(dim=-1).clamp_min(1e-8)
        normal_alignment = (
            surface_normal.unsqueeze(2) * patch_normal.unsqueeze(1)
        ).sum(dim=-1).abs()
        normal_consistency = (assign * normal_alignment).sum(dim=(1, 2)) / usage.sum(
            dim=-1
        ).clamp_min(1e-8)
        token_unit = patch_tokens / patch_tokens.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        token_cos = pt.bmm(token_unit, token_unit.transpose(1, 2))
        offdiag = ~pt.eye(patch_count, device=device, dtype=pt.bool)
        token_diversity = 1.0 - token_cos[:, offdiag].mean(dim=-1)
        residue_mask = (
            pt.arange(max_residues, device=device).unsqueeze(0)
            < pt.as_tensor(residue_counts, device=device).unsqueeze(1)
        )
        residue_count_tensor = pt.as_tensor(
            residue_counts, device=device, dtype=patch_context.dtype
        ).clamp_min(1.0)
        patch_update_norm = (
            patch_context.norm(dim=-1) * residue_mask
        ).sum(dim=-1) / residue_count_tensor
        gate_mask = residue_mask.unsqueeze(-1)
        gate_count = residue_count_tensor * float(gate.shape[-1])
        gate_mean = (gate * gate_mask).sum(dim=(1, 2)) / gate_count
        gate_variance = (
            (gate - gate_mean.view(-1, 1, 1)).square() * gate_mask
        ).sum(dim=(1, 2)) / gate_count
        gate_std = gate_variance.clamp_min(0.0).sqrt()

        outputs = []
        for sample_index, request in enumerate(requests):
            partition = request["connected_partition"]
            diagnostics = {
                "patch_assignment_entropy": entropy[sample_index].detach(),
                "patch_count_effective": assign.new_tensor(float(patch_count)).detach(),
                "patch_usage": usage_norm[sample_index].detach(),
                "patch_usage_active": active[sample_index].detach(),
                "patch_usage_entropy": usage_entropy[sample_index].detach(),
                "patch_spatial_compactness": compactness[sample_index].detach(),
                "patch_normal_consistency": normal_consistency[sample_index].detach(),
                "patch_token_norm_mean": patch_tokens[sample_index].norm(dim=-1).mean().detach(),
                "patch_token_diversity": token_diversity[sample_index].detach(),
                "patch_context_update_norm": patch_update_norm[sample_index].detach(),
                "patch_residue_gate_mean": gate_mean[sample_index].detach(),
                "patch_residue_gate_std": gate_std[sample_index].detach(),
                "patch_tokens": patch_tokens[sample_index],
                "patch_positions": patch_pos[sample_index],
                "patch_assignment": assign[sample_index],
                "patch_adjacency": adjacency[sample_index],
                "patch_surface_graph_idx": partition["surface_graph_idx"].to(device=device),
                "patch_seed_surface_idx": partition["seeds"].to(device=device),
            }
            if replay_surface_gate is not None:
                diagnostics["patch_surface_gate"] = replay_surface_gate[
                    sample_index
                ].detach()
                diagnostics["patch_gate"] = replay_patch_gate[
                    sample_index
                ].squeeze(-1).detach()
            diagnostics.update(
                {
                    key: value.detach()
                    for key, value in partition.get("diagnostics", {}).items()
                }
            )
            primary_surface_idx = assign[sample_index].argmax(dim=0).long()
            diagnostics["patch_token_primary_residue_idx"] = rs_idx[
                sample_index, primary_surface_idx, 0
            ].detach()
            outputs.append(
                (
                    h_surf[sample_index],
                    surf_context_out[sample_index, : residue_counts[sample_index]],
                    diagnostics,
                )
            )
        return outputs

    def forward(
        self,
        h_res,
        h_surf,
        surface_pos,
        surface_normal,
        surface_geom,
        rs_idx,
        rs_weight,
        surf_context=None,
        patch_state=None,
        surface_lappe=None,
        connected_partition=None,
    ):
        if not self.enabled():
            return h_surf, surf_context, {}
        device = h_surf.device
        dtype = h_surf.dtype
        rs_idx = rs_idx.to(device=device, dtype=pt.long)
        rs_weight = rs_weight.to(device=device, dtype=dtype)
        nearest_h = (h_res[rs_idx] * rs_weight.unsqueeze(-1)).sum(dim=1)
        surface_normal = surface_normal.to(device=device, dtype=dtype)
        surface_geom = surface_geom.to(device=device, dtype=dtype)
        lappe = None
        if self.lappe_dim > 0:
            if surface_lappe is None:
                lappe = h_surf.new_zeros((h_surf.shape[0], self.lappe_dim))
            else:
                lappe = surface_lappe.to(device=device, dtype=dtype)
                if lappe.shape[-1] < self.lappe_dim:
                    lappe = pt.cat([lappe, lappe.new_zeros((lappe.shape[0], self.lappe_dim - lappe.shape[-1]))], dim=-1)
                elif lappe.shape[-1] > self.lappe_dim:
                    lappe = lappe[:, : self.lappe_dim]
        if self.connected_mode:
            if connected_partition is None:
                raise ValueError("connected patch mode requires a precomputed connected partition")
            assign = connected_partition["assign"].to(device=device, dtype=dtype)
        else:
            assign_parts = [h_surf, nearest_h, surface_geom, surface_normal]
            if lappe is not None:
                assign_parts.append(lappe)
            assign_logits = self.assignment(pt.cat(assign_parts, dim=-1))
            assign = pt.softmax(assign_logits, dim=-1)
        usage = assign.sum(dim=0).clamp_min(1e-6)
        patch_h = assign.transpose(0, 1).matmul(h_surf) / usage.unsqueeze(-1)
        centered_pos = surface_pos.to(device=device, dtype=dtype) - surface_pos.to(device=device, dtype=dtype).mean(dim=0, keepdim=True)
        patch_pos = assign.transpose(0, 1).matmul(centered_pos) / usage.unsqueeze(-1)
        patch_normal = assign.transpose(0, 1).matmul(surface_normal) / usage.unsqueeze(-1)
        patch_normal = patch_normal / patch_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        patch_geom = assign.transpose(0, 1).matmul(surface_geom) / usage.unsqueeze(-1)
        patch_lappe = None
        if lappe is not None:
            patch_lappe = assign.transpose(0, 1).matmul(lappe) / usage.unsqueeze(-1)
        density = (usage / max(float(assign.shape[0]), 1.0)).unsqueeze(-1)
        token_parts = [patch_h, patch_pos, patch_normal, patch_geom, density]
        if patch_lappe is not None:
            token_parts.append(patch_lappe)
        patch_tokens = self.patch_token(pt.cat(token_parts, dim=-1))
        if patch_state is not None and patch_state.shape == patch_tokens.shape:
            patch_tokens = self.patch_norm(patch_tokens + patch_state.to(device=device, dtype=dtype))
        if self.mode in {"self_attn", "connected_pool_adj"}:
            attention_mask = None
            if self.mode == "connected_pool_adj":
                attention_mask = ~connected_partition["adjacency"].to(device=device, dtype=pt.bool)
            attn_out, _ = self.patch_self_attn(
                patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0),
                attn_mask=attention_mask, need_weights=False,
            )
            patch_tokens = self.patch_norm(patch_tokens + attn_out.squeeze(0))
        if self.patch_importance is None:
            replay_dropout_rng(
                patch_tokens,
                (int(patch_tokens.shape[0]), int(patch_tokens.shape[-1]) // 2),
                probability=float(self.canonical_replay_dropout_probability),
                training=self.training,
            )
            replay_patch_gate = None
            replay_surface_gate = None
        else:
            replay_patch_gate = self.patch_importance(patch_tokens)
            replay_surface_gate = assign.matmul(replay_patch_gate).squeeze(-1)
        patch_gate = replay_patch_gate if self.mode == "gate" else None
        surface_gate = replay_surface_gate if self.mode == "gate" else None
        patch_attn = None
        patch_context = None
        gate = None
        h_surf_out = h_surf
        surf_context_out = surf_context
        if self.mode == "gate":
            h_surf_out = h_surf * (1.0 + self.beta * surface_gate.unsqueeze(-1))
        elif self.mode in {"attn", "self_attn", "replace_attn", "independent_attn", "connected_pool", "connected_pool_adj"} and surf_context is not None:
            ctx, _ = self.residue_patch_attn(
                h_res.unsqueeze(0), patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0),
                need_weights=False,
            )
            if not self.training:
                with pt.no_grad():
                    _unused_ctx, patch_attn = self.residue_patch_attn(
                        h_res.unsqueeze(0), patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0),
                        need_weights=True, average_attn_weights=True,
                    )
            ctx = ctx.squeeze(0)
            gate = self.residue_gate(pt.cat([h_res, ctx], dim=-1))
            patch_context = self.beta * gate * ctx
            if self.mode == "replace_attn":
                surf_context_out = patch_context
            elif self.mode == "independent_attn":
                # Patch tokens form an independent residue update path; they do
                # not rewrite the dense surface tokens, only the residue context.
                surf_context_out = surf_context + patch_context
            else:
                surf_context_out = surf_context + patch_context
        diag = self._summary(
            assign, usage, patch_pos, patch_normal, patch_tokens, centered_pos, surface_normal,
            patch_attn=patch_attn, patch_update=patch_context, residue_gate=gate,
            extra_diagnostics=(connected_partition.get("diagnostics") if connected_partition is not None else None),
        )
        if patch_gate is not None:
            diag["patch_surface_gate"] = surface_gate.detach()
            diag["patch_gate"] = patch_gate.squeeze(-1).detach()
        elif self.connected_mode and replay_surface_gate is not None:
            diag["patch_surface_gate"] = replay_surface_gate.detach()
            diag["patch_gate"] = replay_patch_gate.squeeze(-1).detach()
        diag["patch_tokens"] = patch_tokens
        diag["patch_positions"] = patch_pos
        diag["patch_assignment"] = assign
        if connected_partition is not None:
            diag["patch_adjacency"] = connected_partition["adjacency"].to(device=device)
            diag["patch_surface_graph_idx"] = connected_partition["surface_graph_idx"].to(device=device)
            diag["patch_seed_surface_idx"] = connected_partition["seeds"].to(device=device)
        primary_surface_idx = assign.argmax(dim=0).to(device=device, dtype=pt.long)
        diag["patch_token_primary_residue_idx"] = rs_idx[primary_surface_idx, 0].detach().to(device=device, dtype=pt.long)
        return h_surf_out, surf_context_out, diag


class SurfaceLaplacianSpectralFilter(pt.nn.Module):
    """Low-rank surface operator using cached Laplacian eigenvectors."""

    def __init__(self, dim=128, lappe_dim=16, dropout=0.1):
        super().__init__()
        self.lappe_dim = max(0, int(lappe_dim))
        self.freq_logits = pt.nn.Parameter(pt.zeros(self.lappe_dim))
        self.update_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.update_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.Sigmoid(),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h_surf, surface_lappe):
        if self.lappe_dim <= 0 or surface_lappe is None:
            return h_surf
        u = surface_lappe.to(device=h_surf.device, dtype=h_surf.dtype)
        if u.shape[-1] < self.lappe_dim:
            u = pt.cat([u, u.new_zeros((u.shape[0], self.lappe_dim - u.shape[-1]))], dim=-1)
        elif u.shape[-1] > self.lappe_dim:
            u = u[:, : self.lappe_dim]
        # Normalize columns so U^T H is a stable graph-spectral projection even
        # when eigenvectors come from cached approximations.
        u = u / u.norm(dim=0, keepdim=True).clamp_min(1e-6)
        coeff = u.transpose(0, 1).matmul(h_surf)
        gate = pt.sigmoid(self.freq_logits.to(device=h_surf.device, dtype=h_surf.dtype)).view(-1, 1)
        filtered = u.matmul(coeff * gate)
        update = self.update_proj(filtered)
        mix = self.update_gate(pt.cat([h_surf, update], dim=-1))
        return self.norm(h_surf + mix * update)


class ResidueInternalLaplacianOperator(pt.nn.Module):
    """Learned protein-internal diffusion over the residue contact graph."""

    def __init__(self, dim=128, relation_k=16, layers=2, beta=0.1, dropout=0.1):
        super().__init__()
        self.relation_k = max(2, int(relation_k))
        self.layers = max(1, int(layers))
        self.beta = float(beta)
        self.update_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.gates = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.Sigmoid(),
                )
                for _ in range(self.layers)
            ]
        )
        self.norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(self.layers)])
        self.last_diagnostics = {}

    def _knn(self, centers, graph_cache):
        n = int(centers.shape[0])
        if n <= 1:
            empty_idx = centers.new_empty((n, 0), dtype=pt.long)
            empty_dist = centers.new_empty((n, 0))
            return empty_idx, empty_dist
        if graph_cache:
            idx = graph_cache.get(f"knn_idx_k{self.relation_k}")
            dist = graph_cache.get(f"knn_dist_k{self.relation_k}")
            if isinstance(idx, pt.Tensor) and isinstance(dist, pt.Tensor) and idx.numel() > 0:
                idx = idx.to(device=centers.device, dtype=pt.long)
                dist = dist.to(device=centers.device, dtype=centers.dtype)
                return idx[:, : min(idx.shape[1], n - 1)], dist[:, : min(dist.shape[1], n - 1)]
        dist_mat = pt.cdist(centers, centers).clamp_min(0.0)
        dist_mat = dist_mat + pt.eye(n, device=centers.device, dtype=centers.dtype) * 1e6
        k = min(self.relation_k, n - 1)
        dist, idx = pt.topk(dist_mat, k=k, dim=-1, largest=False)
        return idx, dist

    def forward(self, h_res, centers, graph_cache=None):
        self.last_diagnostics = {}
        if self.beta == 0.0 or h_res.shape[0] <= 1:
            return h_res
        centers = centers.to(device=h_res.device, dtype=h_res.dtype)
        idx, dist = self._knn(centers, graph_cache)
        if idx.shape[1] == 0:
            return h_res
        valid = pt.isfinite(dist)
        weight = pt.softmax(-dist.masked_fill(~valid, 1e6) / dist.detach().median().clamp_min(1.0), dim=-1)
        weight = weight * valid.to(dtype=h_res.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        h = h_res
        diffusion_norms = []
        for update_mlp, gate_mlp, norm in zip(self.update_mlps, self.gates, self.norms):
            neigh = (h[idx] * weight.unsqueeze(-1)).sum(dim=1)
            lap_msg = neigh - h
            update = update_mlp(pt.cat([h, lap_msg], dim=-1))
            gate = gate_mlp(pt.cat([h, lap_msg], dim=-1))
            h = norm(h + self.beta * gate * update)
            diffusion_norms.append(lap_msg.norm(dim=-1).mean().detach())
        if diffusion_norms:
            self.last_diagnostics["internal_operator_diffusion_norm"] = pt.stack(diffusion_norms).mean()
        return h
