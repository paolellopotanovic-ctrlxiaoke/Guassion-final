"""Bidirectional residue/surface fusion and structural patch modules."""

import math

import torch as pt

from .gaussian_projection import inverse_softplus
from .residue_backbone import InterfaceHead


class AttentionSurfaceToResiduePool(pt.nn.Module):
    def __init__(
        self,
        dim=128,
        surface_topk=24,
        dropout=0.1,
        mode="attention",
        gaussian_prior_weight=0.0,
        sigma_tangent=8.0,
        sigma_normal=4.0,
        phys_feature_dim=0,
        phys_attention_bias_weight=0.0,
        phys_context_weight=0.0,
        phys_message_weight=0.0,
        surface_vector_bridge_mode="none",
        surface_vector_dim=16,
    ):
        super().__init__()
        self.surface_topk = int(surface_topk)
        self.mode = str(mode)
        if self.mode not in {"attention", "per_surface_gaussian", "geoparam_attention"}:
            raise ValueError("surface-to-residue mode must be one of: attention, per_surface_gaussian, geoparam_attention")
        self.gaussian_prior_weight = float(gaussian_prior_weight)
        self.sigma_tangent = float(sigma_tangent)
        self.sigma_normal = float(sigma_normal)
        self.phys_feature_dim = int(phys_feature_dim)
        self.phys_attention_bias_weight = float(phys_attention_bias_weight)
        self.phys_context_weight = float(phys_context_weight)
        self.phys_message_weight = float(phys_message_weight)
        self.surface_vector_bridge_mode = str(surface_vector_bridge_mode)
        if self.surface_vector_bridge_mode not in {"none", "invariant", "equivariant"}:
            raise ValueError("surface_vector_bridge_mode must be one of: none, invariant, equivariant")
        self.surface_vector_dim = int(surface_vector_dim)
        vector_bridge_dim = 4 * self.surface_vector_dim if self.surface_vector_bridge_mode != "none" else 0
        score_input_dim = dim * 2 + 5 + vector_bridge_dim
        self.last_residue_vector_update = None
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(score_input_dim),
            pt.nn.Linear(score_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.context_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.surface_sigma_head = None
        if self.mode == "per_surface_gaussian":
            self.surface_sigma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
            last = self.surface_sigma_head[-1]
            pt.nn.init.zeros_(last.weight)
            pt.nn.init.zeros_(last.bias)
        self.geoparam_sigma_head = None
        self.geoparam_gamma_head = None
        if self.mode == "geoparam_attention":
            self.geoparam_sigma_min = 0.5
            self.geoparam_gamma_init = self.gaussian_prior_weight if self.gaussian_prior_weight > 0.0 else 0.25
            self.geoparam_sigma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
            sigma_last = self.geoparam_sigma_head[-1]
            pt.nn.init.zeros_(sigma_last.weight)
            base_sigma = pt.as_tensor([self.sigma_tangent, self.sigma_normal], dtype=pt.float32)
            with pt.no_grad():
                sigma_last.bias.copy_(inverse_softplus(base_sigma - self.geoparam_sigma_min))
            self.geoparam_gamma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(score_input_dim),
                pt.nn.Linear(score_input_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            gamma_last = self.geoparam_gamma_head[-1]
            pt.nn.init.zeros_(gamma_last.weight)
            gamma_init = min(max(float(self.geoparam_gamma_init), 1e-4), 1.0 - 1e-4)
            with pt.no_grad():
                gamma_last.bias.fill_(math.log(gamma_init / (1.0 - gamma_init)))
        if vector_bridge_dim > 0:
            self.vector_context_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + vector_bridge_dim),
                pt.nn.Linear(dim + vector_bridge_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
        else:
            self.vector_context_proj = None
        if self.phys_feature_dim > 0:
            self.phys_pair_score = pt.nn.Sequential(
                pt.nn.LayerNorm(self.phys_feature_dim * 3 + 1),
                pt.nn.Linear(self.phys_feature_dim * 3 + 1, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            self.phys_context_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + self.phys_feature_dim),
                pt.nn.Linear(dim + self.phys_feature_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.phys_edge_message = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + 5 + self.phys_feature_dim * 3 + 1),
                pt.nn.Linear(dim * 2 + 5 + self.phys_feature_dim * 3 + 1, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.phys_message_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.Sigmoid(),
            )
        else:
            self.phys_pair_score = None
            self.phys_context_proj = None
            self.phys_edge_message = None
            self.phys_message_gate = None

    def forward(
        self,
        h_res,
        h_surf,
        residue_centers,
        surface_pos,
        surface_normal,
        residue_phys_features=None,
        surface_phys_features=None,
        surface_vectors=None,
        residue_vectors=None,
        geometry_cache=None,
    ):
        self.last_residue_vector_update = None
        if geometry_cache is None:
            dist = pt.cdist(residue_centers, surface_pos).clamp_min(0.0)
            k = min(self.surface_topk, surface_pos.shape[0])
            idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
            rel = surface_pos[idx] - residue_centers.unsqueeze(1)
            d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            direction = rel / d
            normal = surface_normal[idx]
            normal_proj = (direction * normal).sum(dim=-1, keepdim=True)
            geom = pt.cat([d, direction, normal_proj], dim=-1)
        else:
            idx = geometry_cache["idx"]
            rel = geometry_cache["rel"]
            d = geometry_cache["d"]
            direction = geometry_cache["direction"]
            normal = geometry_cache["normal"]
            geom = geometry_cache["geom"]
        surf = h_surf[idx]
        vector_pair = None
        if self.vector_context_proj is not None and surface_vectors is not None:
            surf_v_all = surface_vectors.to(device=h_res.device, dtype=h_res.dtype)
            if surf_v_all.ndim == 3 and surf_v_all.shape[0] == surface_pos.shape[0]:
                surf_v = surf_v_all[idx]
                v_norm = surf_v.norm(dim=-1)
                v_dir = (surf_v * direction.unsqueeze(-2)).sum(dim=-1)
                v_normal = (surf_v * normal.unsqueeze(-2)).sum(dim=-1)
                vector_pair = pt.cat([v_norm, v_dir, v_dir.abs(), v_normal], dim=-1)
        res_expand = h_res.unsqueeze(1).expand_as(surf)
        score_parts = [res_expand, surf, geom]
        if vector_pair is not None:
            score_parts.append(vector_pair)
        score_input = pt.cat(score_parts, dim=-1)
        if self.mode == "per_surface_gaussian":
            d_normal = (rel * normal).sum(dim=-1).abs()
            d_tangent = (d.squeeze(-1).square() - d_normal.square()).clamp_min(0.0).sqrt()
            raw_sigma = self.surface_sigma_head(h_surf).to(device=h_res.device, dtype=h_res.dtype)
            # Initialize at the old SOTA prior scale: tangent 8A, normal 4A.
            sigma_t = 2.0 + 12.0 * pt.sigmoid(raw_sigma[:, 0])
            sigma_n = 1.0 + 6.0 * pt.sigmoid(raw_sigma[:, 1])
            local_sigma_t = sigma_t[idx].clamp_min(1e-6)
            local_sigma_n = sigma_n[idx].clamp_min(1e-6)
            logits = -0.5 * ((d_tangent / local_sigma_t) ** 2 + (d_normal / local_sigma_n) ** 2)
            self.last_diagnostics = {
                "surface_to_residue_sigma_tangent_mean": sigma_t.mean().detach(),
                "surface_to_residue_sigma_tangent_std": sigma_t.std(unbiased=False).detach(),
                "surface_to_residue_sigma_normal_mean": sigma_n.mean().detach(),
                "surface_to_residue_sigma_normal_std": sigma_n.std(unbiased=False).detach(),
            }
        else:
            logits = self.score(score_input).squeeze(-1)
            self.last_diagnostics = {}
        if self.mode == "geoparam_attention":
            d_normal = (rel * normal).sum(dim=-1).abs()
            d_tangent = (d.squeeze(-1).square() - d_normal.square()).clamp_min(0.0).sqrt()
            sigma_res = self.geoparam_sigma_min + pt.nn.functional.softplus(self.geoparam_sigma_head(h_res))
            sigma_t = sigma_res[:, 0].clamp_min(1e-6)
            sigma_n = sigma_res[:, 1].clamp_min(1e-6)
            prior = -0.5 * (
                (d_tangent / sigma_t.unsqueeze(1)) ** 2
                + (d_normal / sigma_n.unsqueeze(1)) ** 2
            )
            gamma = pt.sigmoid(self.geoparam_gamma_head(score_input)).squeeze(-1)
            logits = logits + gamma * prior
            self.last_diagnostics = {
                "surface_to_residue_geoparam_sigma_tangent_mean": sigma_t.mean().detach(),
                "surface_to_residue_geoparam_sigma_tangent_min": sigma_t.min().detach(),
                "surface_to_residue_geoparam_sigma_tangent_max": sigma_t.max().detach(),
                "surface_to_residue_geoparam_sigma_tangent_std": sigma_t.std(unbiased=False).detach(),
                "surface_to_residue_geoparam_sigma_normal_mean": sigma_n.mean().detach(),
                "surface_to_residue_geoparam_sigma_normal_min": sigma_n.min().detach(),
                "surface_to_residue_geoparam_sigma_normal_max": sigma_n.max().detach(),
                "surface_to_residue_geoparam_sigma_normal_std": sigma_n.std(unbiased=False).detach(),
                "surface_to_residue_geoparam_gamma_mean": gamma.mean().detach(),
                "surface_to_residue_geoparam_gamma_std": gamma.std(unbiased=False).detach(),
                "surface_to_residue_geoparam_prior_mean": prior.mean().detach(),
                "surface_to_residue_geoparam_prior_std": prior.std(unbiased=False).detach(),
            }
        if self.mode == "attention" and self.gaussian_prior_weight > 0:
            d_normal = (rel * normal).sum(dim=-1).abs()
            d_tangent = (d.squeeze(-1).square() - d_normal.square()).clamp_min(0.0).sqrt()
            prior = -0.5 * (
                (d_tangent / max(self.sigma_tangent, 1e-6)) ** 2
                + (d_normal / max(self.sigma_normal, 1e-6)) ** 2
            )
            logits = logits + self.gaussian_prior_weight * prior
        use_phys = (
            self.phys_pair_score is not None
            and self.phys_attention_bias_weight != 0.0
            and residue_phys_features is not None
            and surface_phys_features is not None
        )
        surf_phys = None
        if use_phys:
            res_phys = residue_phys_features.to(device=h_res.device, dtype=h_res.dtype)
            surf_phys_all = surface_phys_features.to(device=h_res.device, dtype=h_res.dtype)
            surf_phys = surf_phys_all[idx]
            res_phys_pair = res_phys.unsqueeze(1).expand_as(surf_phys)
            denom = res_phys_pair.norm(dim=-1, keepdim=True).clamp_min(1e-6) * surf_phys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            phys_cos = (res_phys_pair * surf_phys).sum(dim=-1, keepdim=True) / denom
            phys_pair = pt.cat([res_phys_pair, surf_phys, (res_phys_pair - surf_phys).abs(), phys_cos], dim=-1)
            logits = logits + self.phys_attention_bias_weight * self.phys_pair_score(phys_pair).squeeze(-1)
        else:
            phys_pair = None
        attn = pt.softmax(logits, dim=-1)
        ctx = (surf * attn.unsqueeze(-1)).sum(dim=1)
        out = self.context_proj(ctx)
        if self.vector_context_proj is not None and vector_pair is not None:
            vector_ctx = (vector_pair * attn.unsqueeze(-1)).sum(dim=1)
            out = out + self.vector_context_proj(pt.cat([ctx, vector_ctx], dim=-1))
            if self.surface_vector_bridge_mode == "equivariant" and residue_vectors is not None:
                raw_vector_ctx = (surf_v * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
                if raw_vector_ctx.shape == residue_vectors.shape:
                    self.last_residue_vector_update = raw_vector_ctx
        if (
            self.phys_context_proj is not None
            and self.phys_context_weight != 0.0
            and residue_phys_features is not None
            and surface_phys_features is not None
        ):
            if surf_phys is None:
                surf_phys = surface_phys_features.to(device=h_res.device, dtype=h_res.dtype)[idx]
            phys_ctx = (surf_phys * attn.unsqueeze(-1)).sum(dim=1)
            out = out + self.phys_context_weight * self.phys_context_proj(pt.cat([ctx, phys_ctx], dim=-1))
        if (
            self.phys_edge_message is not None
            and self.phys_message_weight != 0.0
            and residue_phys_features is not None
            and surface_phys_features is not None
        ):
            if surf_phys is None:
                surf_phys = surface_phys_features.to(device=h_res.device, dtype=h_res.dtype)[idx]
            if phys_pair is None:
                res_phys = residue_phys_features.to(device=h_res.device, dtype=h_res.dtype)
                res_phys_pair = res_phys.unsqueeze(1).expand_as(surf_phys)
                denom = res_phys_pair.norm(dim=-1, keepdim=True).clamp_min(1e-6) * surf_phys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                phys_cos = (res_phys_pair * surf_phys).sum(dim=-1, keepdim=True) / denom
                phys_pair = pt.cat([res_phys_pair, surf_phys, (res_phys_pair - surf_phys).abs(), phys_cos], dim=-1)
            edge_msg = self.phys_edge_message(pt.cat([res_expand, surf, geom, phys_pair], dim=-1))
            msg_ctx = (edge_msg * attn.unsqueeze(-1)).sum(dim=1)
            msg_gate = self.phys_message_gate(pt.cat([h_res, msg_ctx], dim=-1))
            out = out + self.phys_message_weight * msg_gate * msg_ctx
        return out, attn

class AttentionResidueToSurfacePool(pt.nn.Module):
    """Residue-to-surface cross update used by bidirectional co-update blocks."""

    def __init__(self, dim=128, residue_topk=16, dropout=0.1):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2 + 5),
            pt.nn.Linear(dim * 2 + 5, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.context_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )

    def forward(self, h_surf, h_res, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        res = h_res[idx]
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / d
        normal = surface_normal.unsqueeze(1).expand(-1, k, -1)
        normal_proj = (direction * normal).sum(dim=-1, keepdim=True)
        geom = pt.cat([d, direction, normal_proj], dim=-1)
        surf_expand = h_surf.unsqueeze(1).expand_as(res)
        logits = self.score(pt.cat([surf_expand, res, geom], dim=-1)).squeeze(-1)
        attn = pt.softmax(logits, dim=-1)
        ctx = (res * attn.unsqueeze(-1)).sum(dim=1)
        return self.context_proj(ctx), attn


class SurfaceBusFromViews(pt.nn.Module):
    """Surface-point arbitration over view-specific residue states."""

    def __init__(self, dim=128, view_count=4, residue_topk=16, dropout=0.1):
        super().__init__()
        self.view_count = max(1, int(view_count))
        self.readers = pt.nn.ModuleList(
            [
                AttentionResidueToSurfacePool(dim=dim, residue_topk=residue_topk, dropout=dropout)
                for _ in range(self.view_count)
            ]
        )
        self.view_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * self.view_count + 6),
            pt.nn.Linear(dim * self.view_count + 6, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.view_count),
        )
        self.out_norm = pt.nn.LayerNorm(dim)
        self.last_diagnostics = {}

    def forward(self, h_surf_base, view_stack, surface_pos, surface_normal, surface_geom, residue_centers):
        if view_stack is None or view_stack.numel() == 0:
            self.last_diagnostics = {}
            return h_surf_base
        usable_views = min(int(view_stack.shape[0]), self.view_count)
        view_msgs = []
        attn_entropies = []
        for view_idx in range(usable_views):
            msg, attn = self.readers[view_idx](
                h_surf_base,
                view_stack[view_idx],
                surface_pos,
                surface_normal,
                residue_centers,
            )
            view_msgs.append(msg)
            attn_entropies.append(-(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean())
        while len(view_msgs) < self.view_count:
            view_msgs.append(h_surf_base.new_zeros(h_surf_base.shape))
        stacked = pt.stack(view_msgs, dim=1)
        gate_logits = self.view_gate(pt.cat([stacked.reshape(stacked.shape[0], -1), surface_geom], dim=-1))
        if usable_views < self.view_count:
            gate_logits[:, usable_views:] = -1.0e4
        gate = pt.softmax(gate_logits, dim=-1)
        bus = (gate.unsqueeze(-1) * stacked).sum(dim=1)
        entropy = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_diagnostics = {
            "surface_bus_view_entropy": entropy.detach(),
            "surface_bus_view_max": gate.max(dim=-1).values.mean().detach(),
            "surface_bus_update_norm": bus.norm(dim=-1).mean().detach(),
            "surface_bus_view_count": h_surf_base.new_tensor(float(usable_views)).detach(),
        }
        if attn_entropies:
            self.last_diagnostics["surface_bus_residue_entropy"] = pt.stack(attn_entropies).mean().detach()
        return self.out_norm(bus)


class StructuralCommunityDynamicAttention(pt.nn.Module):
    """Dynamic structural-community routing between residues and surface tokens."""

    def __init__(self, dim=128, num_communities=16, geom_bias=0.25, dropout=0.1):
        super().__init__()
        self.num_communities = max(1, int(num_communities))
        self.geom_bias = float(geom_bias)
        self.surface_score = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2 + 5),
            pt.nn.Linear(dim * 2 + 5, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.surface_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.surface_gate = pt.nn.Sequential(pt.nn.LayerNorm(dim * 2), pt.nn.Linear(dim * 2, dim), pt.nn.Sigmoid())
        self.residue_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.residue_gate = pt.nn.Sequential(pt.nn.LayerNorm(dim * 2), pt.nn.Linear(dim * 2, dim), pt.nn.Sigmoid())
        self.surface_norm = pt.nn.LayerNorm(dim)
        self.residue_norm = pt.nn.LayerNorm(dim)
        self.last_diagnostics = {}

    def _community_ids(self, centers):
        n = int(centers.shape[0])
        c = min(self.num_communities, max(n, 1))
        if n <= c:
            return pt.arange(n, device=centers.device, dtype=pt.long), c
        center = centers.mean(dim=0, keepdim=True)
        first = (centers - center).norm(dim=-1).argmax()
        seeds = [first]
        min_dist = pt.cdist(centers, centers[first : first + 1]).squeeze(-1)
        for _ in range(1, c):
            nxt = min_dist.argmax()
            seeds.append(nxt)
            dist_new = pt.cdist(centers, centers[nxt : nxt + 1]).squeeze(-1)
            min_dist = pt.minimum(min_dist, dist_new)
        seed_idx = pt.stack(seeds).to(device=centers.device, dtype=pt.long)
        dist_seed = pt.cdist(centers, centers[seed_idx])
        return dist_seed.argmin(dim=-1).to(dtype=pt.long), c

    def forward(self, h_res, h_surf, residue_centers, surface_pos, surface_normal):
        if h_res.shape[0] == 0 or h_surf.shape[0] == 0:
            self.last_diagnostics = {}
            return h_res, h_surf
        centers = residue_centers.to(device=h_res.device, dtype=h_res.dtype)
        ids, count = self._community_ids(centers)
        comm_tokens = []
        comm_centers = []
        active = []
        for cid in range(count):
            mask = ids == cid
            active.append(mask.to(dtype=h_res.dtype).sum())
            if pt.any(mask):
                comm_tokens.append(h_res[mask].mean(dim=0))
                comm_centers.append(centers[mask].mean(dim=0))
            else:
                comm_tokens.append(h_res.new_zeros(h_res.shape[-1]))
                comm_centers.append(centers.mean(dim=0))
        comm = pt.stack(comm_tokens, dim=0)
        comm_pos = pt.stack(comm_centers, dim=0)
        rel = comm_pos.unsqueeze(0) - surface_pos.unsqueeze(1)
        d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / d
        normal = surface_normal.unsqueeze(1).expand(-1, count, -1)
        normal_proj = (direction * normal).sum(dim=-1, keepdim=True)
        geom = pt.cat([d, direction, normal_proj], dim=-1)
        surf_expand = h_surf.unsqueeze(1).expand(-1, count, -1)
        comm_expand = comm.unsqueeze(0).expand(h_surf.shape[0], -1, -1)
        logits = self.surface_score(pt.cat([surf_expand, comm_expand, geom], dim=-1)).squeeze(-1)
        if self.geom_bias != 0.0:
            logits = logits - self.geom_bias * d.squeeze(-1) / d.detach().mean().clamp_min(1e-6)
        attn = pt.softmax(logits, dim=-1)
        surf_ctx = (attn.unsqueeze(-1) * comm_expand).sum(dim=1)
        surf_input = pt.cat([h_surf, surf_ctx], dim=-1)
        h_surf_out = self.surface_norm(h_surf + self.surface_gate(surf_input) * self.surface_update(surf_input))
        res_ctx = comm[ids]
        res_input = pt.cat([h_res, res_ctx], dim=-1)
        h_res_out = self.residue_norm(h_res + self.residue_gate(res_input) * self.residue_update(res_input))
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        active_t = pt.stack(active) if active else h_res.new_zeros(1)
        self.last_diagnostics = {
            "struct_comm_active_mean": (active_t > 0).to(dtype=h_res.dtype).sum().detach(),
            "struct_comm_surface_entropy": entropy.detach(),
            "struct_comm_surface_gate_mean": self.surface_gate(surf_input).detach().mean(),
            "struct_comm_residue_gate_mean": self.residue_gate(res_input).detach().mean(),
        }
        return h_res_out, h_surf_out


class MotifPatchCrossAttention(pt.nn.Module):
    """Residue motif tokens interacting with existing surface patch tokens."""

    def __init__(self, dim=128, num_motifs=32, mode="dual_cross", beta=0.1, heads=4, dropout=0.1, geom_bias=0.1):
        super().__init__()
        self.num_motifs = max(1, int(num_motifs))
        self.mode = str(mode)
        if self.mode not in {"feedback", "dual_cross"}:
            raise ValueError("motif_patch_mode must be one of: none, feedback, dual_cross")
        self.beta = float(beta)
        self.geom_bias = float(geom_bias)
        heads = max(1, int(heads))
        if dim % heads != 0:
            heads = 1
        self.motif_self_attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.motif_from_patch = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.patch_from_motif = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.residue_from_patch = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.motif_norm = pt.nn.LayerNorm(dim)
        self.patch_norm = pt.nn.LayerNorm(dim)
        self.residue_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.residue_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.context_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.context_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.residue_norm = pt.nn.LayerNorm(dim)
        self.context_norm = pt.nn.LayerNorm(dim)
        self.motif_aux_head = InterfaceHead(dim=dim, dropout=dropout)
        self.patch_aux_head = InterfaceHead(dim=dim, dropout=dropout)
        self.last_diagnostics = {}

    def _motif_ids(self, centers):
        n = int(centers.shape[0])
        c = min(self.num_motifs, max(n, 1))
        if n <= c:
            return pt.arange(n, device=centers.device, dtype=pt.long), c
        center = centers.mean(dim=0, keepdim=True)
        first = (centers - center).norm(dim=-1).argmax()
        seeds = [first]
        min_dist = pt.cdist(centers, centers[first : first + 1]).squeeze(-1)
        for _ in range(1, c):
            nxt = min_dist.argmax()
            seeds.append(nxt)
            dist_new = pt.cdist(centers, centers[nxt : nxt + 1]).squeeze(-1)
            min_dist = pt.minimum(min_dist, dist_new)
        seed_idx = pt.stack(seeds).to(device=centers.device, dtype=pt.long)
        dist_seed = pt.cdist(centers, centers[seed_idx])
        return dist_seed.argmin(dim=-1).to(dtype=pt.long), c

    def forward(self, h_res, centers, patch_tokens, surf_context=None):
        self.last_diagnostics = {}
        if patch_tokens is None or patch_tokens.numel() == 0 or h_res.shape[0] == 0:
            return h_res, surf_context, {}
        centers = centers.to(device=h_res.device, dtype=h_res.dtype)
        patch_tokens = patch_tokens.to(device=h_res.device, dtype=h_res.dtype)
        ids, count = self._motif_ids(centers)
        motif_tokens = []
        motif_centers = []
        active = []
        primary = []
        for cid in range(count):
            mask = ids == cid
            active.append(mask.to(dtype=h_res.dtype).sum())
            if pt.any(mask):
                motif_tokens.append(h_res[mask].mean(dim=0))
                motif_centers.append(centers[mask].mean(dim=0))
                local_idx = pt.nonzero(mask, as_tuple=False).squeeze(-1)
                primary.append(local_idx[0])
            else:
                motif_tokens.append(h_res.new_zeros(h_res.shape[-1]))
                motif_centers.append(centers.mean(dim=0))
                primary.append(h_res.new_tensor(0, dtype=pt.long))
        motif = pt.stack(motif_tokens, dim=0)
        motif_pos = pt.stack(motif_centers, dim=0)
        primary_idx = pt.stack(primary).to(device=h_res.device, dtype=pt.long)
        motif_self, _ = self.motif_self_attn(motif.unsqueeze(0), motif.unsqueeze(0), motif.unsqueeze(0), need_weights=False)
        motif = self.motif_norm(motif + motif_self.squeeze(0))
        motif_ctx, motif_attn = self.motif_from_patch(
            motif.unsqueeze(0),
            patch_tokens.unsqueeze(0),
            patch_tokens.unsqueeze(0),
            need_weights=True,
            average_attn_weights=True,
        )
        motif = self.motif_norm(motif + self.beta * motif_ctx.squeeze(0))
        patch_updated = patch_tokens
        patch_attn_entropy = h_res.new_tensor(0.0)
        if self.mode == "dual_cross":
            patch_ctx, patch_attn = self.patch_from_motif(
                patch_tokens.unsqueeze(0),
                motif.unsqueeze(0),
                motif.unsqueeze(0),
                need_weights=True,
                average_attn_weights=True,
            )
            patch_updated = self.patch_norm(patch_tokens + self.beta * patch_ctx.squeeze(0))
            patch_attn_entropy = -(
                patch_attn.squeeze(0).clamp_min(1e-8) * patch_attn.squeeze(0).clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
        patch_res_ctx, patch_res_attn = self.residue_from_patch(
            h_res.unsqueeze(0),
            patch_updated.unsqueeze(0),
            patch_updated.unsqueeze(0),
            need_weights=True,
            average_attn_weights=True,
        )
        patch_res_ctx = patch_res_ctx.squeeze(0)
        motif_res_ctx = motif[ids]
        res_input = pt.cat([h_res, motif_res_ctx, patch_res_ctx], dim=-1)
        res_update = self.residue_update(res_input)
        res_gate = self.residue_gate(res_input)
        h_res_out = self.residue_norm(h_res + self.beta * res_gate * res_update)
        surf_context_out = surf_context
        if surf_context is not None:
            ctx_input = pt.cat([surf_context, motif_res_ctx, patch_res_ctx], dim=-1)
            ctx_update = self.context_update(ctx_input)
            ctx_gate = self.context_gate(ctx_input)
            surf_context_out = self.context_norm(surf_context + self.beta * ctx_gate * ctx_update)
        active_t = pt.stack(active) if active else h_res.new_zeros(1)
        motif_attn_entropy = -(
            motif_attn.squeeze(0).clamp_min(1e-8) * motif_attn.squeeze(0).clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        patch_res_entropy = -(
            patch_res_attn.squeeze(0).clamp_min(1e-8) * patch_res_attn.squeeze(0).clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        diag = {
            "motif_patch_motif_logits": self.motif_aux_head(motif),
            "motif_patch_motif_primary_idx": primary_idx,
            "motif_patch_patch_logits": self.patch_aux_head(patch_updated),
            "motif_patch_motif_count": h_res.new_tensor(float(count)).detach(),
            "motif_patch_active_mean": active_t.detach().float().mean(),
            "motif_patch_motif_patch_entropy": motif_attn_entropy.detach(),
            "motif_patch_patch_motif_entropy": patch_attn_entropy.detach(),
            "motif_patch_residue_patch_entropy": patch_res_entropy.detach(),
            "motif_patch_residue_gate_mean": res_gate.detach().mean(),
        }
        if surf_context is not None:
            diag["motif_patch_context_gate_mean"] = ctx_gate.detach().mean()
        self.last_diagnostics = {k: v for k, v in diag.items() if not k.endswith("_logits") and not k.endswith("_idx")}
        return h_res_out, surf_context_out, diag
