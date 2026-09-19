"""Connected semantic surface-patch prediction head."""

import math

import torch as pt

from .runtime import replay_dropout_rng


class SemanticConnectedPatchHead(pt.nn.Module):
    MODES = {
        "none",
        "amplitude",
        "gaussian",
        "setpatch",
        "full",
        "sparse_probe",
        "sparse_noisyor",
        "graph_motif",
        "graph_motif_refine",
    }

    def __init__(
        self,
        dim=128,
        query_count=8,
        mode="none",
        beta=0.03,
        radius_min=3.0,
        radius_max=16.0,
        adjacency_weight=0.25,
        dropout=0.1,
        attention_heads=4,
        active_topk=4,
        graph_hops=2,
        geodesic_max_steps=96,
        geodesic_normal_penalty=1.0,
        geodesic_tangent_penalty=1.0,
        detach_backbone=True,
        boundary_mix=0.1,
        boundary_temperature=0.7,
    ):
        super().__init__()
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(f"semantic patch mode must be one of {sorted(self.MODES)}, got {self.mode}")
        self.query_count = max(1, int(query_count))
        self.beta_max = max(0.0, float(beta))
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        if self.radius_max <= self.radius_min:
            raise ValueError("semantic patch radius_max must exceed radius_min")
        self.adjacency_weight = float(adjacency_weight)
        self.active_topk = max(1, int(active_topk))
        self.graph_hops = max(1, int(graph_hops))
        self.geodesic_max_steps = max(1, int(geodesic_max_steps))
        self.geodesic_normal_penalty = max(0.0, float(geodesic_normal_penalty))
        self.geodesic_tangent_penalty = max(0.0, float(geodesic_tangent_penalty))
        self.detach_backbone = bool(detach_backbone)
        self.boundary_mix = min(max(float(boundary_mix), 0.0), 0.5)
        self.boundary_temperature = max(float(boundary_temperature), 1e-3)
        self.register_buffer("boundary_refinement_scale", pt.ones(()), persistent=False)

        self.region_amplitude = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )
        self.region_existence = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )
        self.region_radius = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )
        self.residual_scale = pt.nn.Parameter(pt.zeros(()))

        heads = max(1, int(attention_heads))
        if dim % heads != 0:
            heads = 1
        self.patch_queries = pt.nn.Parameter(pt.randn(self.query_count, dim) / math.sqrt(dim))
        self.query_norm = pt.nn.LayerNorm(dim)
        self.query_proj = pt.nn.Linear(dim, dim, bias=False)
        self.region_proj = pt.nn.Linear(dim, dim, bias=False)
        self.surface_proj = pt.nn.Linear(dim, dim, bias=False)
        self.patch_existence = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, 1))

    @property
    def enabled(self):
        return self.mode != "none"

    @property
    def uses_region_supervision(self):
        return self.mode in {
            "amplitude",
            "gaussian",
            "full",
            "sparse_probe",
            "sparse_noisyor",
            "graph_motif",
            "graph_motif_refine",
        }

    @property
    def uses_set_prediction(self):
        return self.mode in {"setpatch", "full", "graph_motif", "graph_motif_refine"}

    def set_boundary_refinement_scale(self, scale):
        self.boundary_refinement_scale.fill_(min(max(float(scale), 0.0), 1.0))

    @staticmethod
    def _surface_to_residue_matrix(assign, rs_idx, rs_weight, residue_count):
        point_count = int(assign.shape[0])
        matrix = assign.new_zeros((point_count, residue_count))
        valid = (rs_idx >= 0) & (rs_idx < residue_count)
        rows = pt.arange(point_count, device=assign.device, dtype=pt.long).unsqueeze(1).expand_as(rs_idx)
        matrix.index_put_((rows[valid], rs_idx[valid]), rs_weight[valid], accumulate=True)
        return matrix

    @staticmethod
    def _region_to_residue(assign, surface_to_residue):
        region_to_residue = assign.transpose(0, 1).matmul(surface_to_residue)
        return region_to_residue / region_to_residue.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _undirected_edges(graph_idx, point_count):
        graph_idx = graph_idx.to(dtype=pt.long)
        rows = pt.arange(point_count, device=graph_idx.device, dtype=pt.long).unsqueeze(1).expand_as(graph_idx)
        valid = (graph_idx >= 0) & (graph_idx < point_count) & (graph_idx != rows)
        source = pt.cat([rows[valid], graph_idx[valid]], dim=0)
        target = pt.cat([graph_idx[valid], rows[valid]], dim=0)
        edge_code = pt.unique(source * point_count + target, sorted=False)
        return edge_code.div(point_count, rounding_mode="floor"), edge_code.remainder(point_count)

    def _graph_geodesic_distances(self, surface_pos, surface_normal, graph_idx, patch_pos):
        point_count = int(surface_pos.shape[0])
        with pt.no_grad(), pt.autocast(device_type=surface_pos.device.type, enabled=False):
            position = surface_pos.float()
            normal = surface_normal.float()
            source, target = self._undirected_edges(graph_idx, point_count)
            delta = position[target] - position[source]
            chord = delta.norm(dim=-1).clamp_min(1e-4)
            normal_bend = 0.5 * (
                1.0 - (normal[source] * normal[target]).sum(dim=-1).clamp(-1.0, 1.0)
            )
            mean_normal = normal[source] + normal[target]
            mean_normal = mean_normal / mean_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            through_surface = (delta * mean_normal).sum(dim=-1).abs() / chord
            edge_weight = chord * (
                1.0
                + self.geodesic_normal_penalty * normal_bend
                + self.geodesic_tangent_penalty * through_surface
            )
            absolute_patch_pos = patch_pos.float() + position.mean(dim=0, keepdim=True)
            center_index = pt.cdist(absolute_patch_pos, position).argmin(dim=-1)
            patch_count = int(center_index.numel())
            distance = position.new_full((patch_count, point_count), float("inf"))
            distance[pt.arange(patch_count, device=position.device), center_index] = 0.0
            target_index = target.unsqueeze(0).expand(patch_count, -1)
            for _ in range(self.geodesic_max_steps):
                candidate = distance[:, source] + edge_weight.unsqueeze(0)
                best = distance.new_full(distance.shape, float("inf"))
                best.scatter_reduce_(1, target_index, candidate, reduce="amin", include_self=True)
                updated = pt.minimum(distance, best)
                if bool(pt.all(updated >= distance - 1e-6)):
                    distance = updated
                    break
                distance = updated
            finite = pt.isfinite(distance)
            if pt.any(~finite):
                euclidean = pt.cdist(position[center_index], position)
                finite_max = distance[finite].max() if pt.any(finite) else euclidean.new_tensor(1.0)
                distance = pt.where(finite, distance, finite_max + euclidean)
        return distance.to(dtype=surface_pos.dtype), center_index

    def _local_boundary_refinement(self, assign, h_surf, patch_tokens, graph_idx):
        labels = assign.argmax(dim=-1)
        point_count, patch_count = assign.shape
        graph_idx = graph_idx.to(device=assign.device, dtype=pt.long)
        rows = pt.arange(point_count, device=assign.device).unsqueeze(1).expand_as(graph_idx)
        valid = (graph_idx >= 0) & (graph_idx < point_count)
        neighbor_labels = labels[graph_idx.clamp(0, max(point_count - 1, 0))]
        candidate = pt.zeros((point_count, patch_count), device=assign.device, dtype=pt.bool)
        candidate.scatter_(1, labels.unsqueeze(-1), True)
        candidate[rows[valid], neighbor_labels[valid]] = True
        boundary = valid.logical_and(neighbor_labels != labels.unsqueeze(-1)).any(dim=-1)
        scale = math.sqrt(max(int(h_surf.shape[-1]), 1))
        score = self.surface_proj(h_surf).matmul(self.region_proj(patch_tokens).transpose(0, 1)) / scale
        score = score.masked_fill(~candidate, -1e4)
        soft_assign = pt.softmax(score / self.boundary_temperature, dim=-1)
        mix = self.boundary_mix * self.boundary_refinement_scale.to(assign)
        refined_boundary = (1.0 - mix) * assign + mix * soft_assign
        refined = pt.where(boundary.unsqueeze(-1), refined_boundary, assign)
        refined = refined / refined.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if pt.any(valid):
            edge_delta = (refined[rows[valid]] - refined[graph_idx[valid]]).abs().mean(dim=-1)
            boundary_tv = edge_delta.mean()
        else:
            boundary_tv = refined.sum() * 0.0
        diagnostics = {
            "semantic_boundary_fraction": boundary.float().mean(),
            "semantic_boundary_shift": (refined - assign).abs().sum(dim=-1).mean(),
            "semantic_boundary_tv": boundary_tv,
        }
        return refined, diagnostics

    @staticmethod
    def _centered_logit(probability):
        logit = pt.logit(probability.clamp(1e-4, 1.0 - 1e-4))
        return logit - logit.mean()

    def _sparse_region_noisy_or(
        self,
        centers,
        surface_pos,
        geodesic_distance,
        amplitude_logits,
        existence_logits,
        radius,
    ):
        nearest_surface = pt.cdist(centers.float(), surface_pos.float()).argmin(dim=-1)
        residue_distance = geodesic_distance[:, nearest_surface].transpose(0, 1).float()
        radius_float = radius.float().unsqueeze(0).clamp_min(1e-6)
        kernel = pt.exp(-0.5 * (residue_distance / radius_float).square())
        amplitude = pt.sigmoid(amplitude_logits.float())
        existence = pt.sigmoid(existence_logits.float())
        active_score = amplitude * existence
        active_count = min(self.active_topk, int(active_score.numel()))
        active_index = pt.topk(active_score, k=active_count, largest=True).indices
        active_mask = pt.zeros_like(active_score)
        active_mask[active_index] = 1.0
        local_probability = kernel * (active_score * active_mask).unsqueeze(0)
        probability = 1.0 - pt.exp(pt.log1p(-local_probability.clamp(max=1.0 - 1e-5)).sum(dim=-1))
        diagnostics = {
            "semantic_active_region_count": active_mask.sum(),
            "semantic_active_region_mass": (active_score * active_mask).sum(),
            "semantic_region_existence_mean": existence.mean(),
            "semantic_region_amplitude_mean": amplitude.mean(),
        }
        return probability.to(dtype=centers.dtype), active_mask, diagnostics

    def _graph_constrained_motifs(
        self,
        patch_tokens,
        assign,
        adjacency,
        surface_to_residue,
        amplitude_logits,
        region_existence_logits,
    ):
        query = self.patch_queries
        scale = math.sqrt(max(int(query.shape[-1]), 1))
        score = self.query_proj(query).matmul(self.region_proj(patch_tokens).transpose(0, 1)) / scale
        seed_soft = pt.softmax(score, dim=-1)
        seed_hard = pt.nn.functional.one_hot(
            seed_soft.argmax(dim=-1), num_classes=int(patch_tokens.shape[0])
        ).to(dtype=seed_soft.dtype)
        seed = seed_hard + seed_soft - seed_soft.detach()
        compatibility = pt.sigmoid(score)
        adjacency_float = adjacency.to(device=score.device, dtype=score.dtype)
        adjacency_float.fill_diagonal_(True)
        transition = adjacency_float / adjacency_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        membership = seed
        for _ in range(self.graph_hops):
            frontier = membership.matmul(transition) * compatibility
            membership = 1.0 - (1.0 - membership) * (1.0 - frontier.clamp(0.0, 1.0))
        region_activity = pt.sigmoid(amplitude_logits) * pt.sigmoid(region_existence_logits)
        functional_membership = membership * region_activity.unsqueeze(0)
        query_context = membership.matmul(patch_tokens) / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        query_state = self.query_norm(query + query_context)
        existence_logits = self.patch_existence(query_state).squeeze(-1)
        surface_masks = functional_membership.matmul(assign.transpose(0, 1)).clamp(1e-4, 1.0 - 1e-4)
        residue_masks = surface_masks.matmul(surface_to_residue)
        residue_support = surface_to_residue.sum(dim=0, keepdim=True).clamp_min(1e-8)
        residue_masks = (residue_masks / residue_support).clamp(1e-4, 1.0 - 1e-4)
        active_probability = pt.sigmoid(existence_logits).unsqueeze(-1) * residue_masks
        union_probability = 1.0 - pt.exp(
            pt.log1p(-active_probability.clamp(max=1.0 - 1e-5)).sum(dim=0)
        )
        return {
            "queries": query_state,
            "region_weights": membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8),
            "surface_masks": surface_masks,
            "residue_masks": residue_masks,
            "existence_logits": existence_logits,
            "probability": union_probability,
            "seed_region": seed_hard.argmax(dim=-1),
        }

    def forward_batched(self, requests):
        """Batch graph-motif heads and pad only the residue dimension."""

        if self.mode not in {"graph_motif", "graph_motif_refine"}:
            return [self.forward(**request) for request in requests]
        surface_counts = {int(request["h_surf"].shape[0]) for request in requests}
        patch_counts = {
            int(request["patch_tokens"].shape[0]) for request in requests
        }
        if len(surface_counts) != 1 or len(patch_counts) != 1:
            return [self.forward(**request) for request in requests]

        batch_size = len(requests)
        device = requests[0]["h_surf"].device
        dtype = requests[0]["h_surf"].dtype
        residue_counts = [int(request["h_res"].shape[0]) for request in requests]
        max_residues = max(residue_counts)
        surface_count = surface_counts.pop()
        patch_count = patch_counts.pop()
        h_res = pt.nn.utils.rnn.pad_sequence(
            [request["h_res"] for request in requests], batch_first=True
        )
        centers = pt.nn.utils.rnn.pad_sequence(
            [request["centers"] for request in requests], batch_first=True
        )
        h_surf = pt.stack([request["h_surf"] for request in requests])
        graph_idx = pt.stack(
            [
                request["surface_graph_idx"].to(device=device, dtype=pt.long)
                for request in requests
            ]
        )
        patch_tokens = pt.stack([request["patch_tokens"] for request in requests])
        assign = pt.stack([request["assign"] for request in requests])
        adjacency = pt.stack(
            [
                request["adjacency"].to(device=device, dtype=pt.bool)
                for request in requests
            ]
        )
        rs_idx = pt.stack(
            [request["rs_idx"].to(device=device, dtype=pt.long) for request in requests]
        )
        rs_weight = pt.stack(
            [request["rs_weight"].to(device=device, dtype=dtype) for request in requests]
        )

        semantic_tokens = patch_tokens.detach() if self.detach_backbone else patch_tokens
        semantic_surface = h_surf.detach() if self.detach_backbone else h_surf
        semantic_assign = assign
        boundary_diagnostics = {}
        if self.mode == "graph_motif_refine":
            labels = assign.argmax(dim=-1)
            valid = (graph_idx >= 0) & (graph_idx < surface_count)
            graph_safe = graph_idx.clamp(0, max(surface_count - 1, 0))
            batch_index = pt.arange(batch_size, device=device).view(batch_size, 1, 1)
            neighbor_labels = labels[batch_index, graph_safe]
            candidate_count = pt.zeros(
                (batch_size, surface_count, patch_count),
                device=device,
                dtype=pt.int32,
            )
            candidate_count.scatter_add_(
                2,
                labels.unsqueeze(-1),
                pt.ones_like(labels.unsqueeze(-1), dtype=pt.int32),
            )
            candidate_count.scatter_add_(
                2, neighbor_labels, valid.to(dtype=pt.int32)
            )
            candidate = candidate_count > 0
            boundary = valid.logical_and(
                neighbor_labels != labels.unsqueeze(-1)
            ).any(dim=-1)
            scale = math.sqrt(max(int(h_surf.shape[-1]), 1))
            score = pt.bmm(
                self.surface_proj(semantic_surface),
                self.region_proj(semantic_tokens).transpose(1, 2),
            ) / scale
            score = score.masked_fill(~candidate, -1e4)
            soft_assign = pt.softmax(score / self.boundary_temperature, dim=-1)
            mix = self.boundary_mix * self.boundary_refinement_scale.to(assign)
            refined_boundary = (1.0 - mix) * assign + mix * soft_assign
            semantic_assign = pt.where(
                boundary.unsqueeze(-1), refined_boundary, assign
            )
            semantic_assign = semantic_assign / semantic_assign.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            neighbor_assign = semantic_assign[batch_index, graph_safe]
            edge_delta = (
                semantic_assign.unsqueeze(2) - neighbor_assign
            ).abs().mean(dim=-1)
            valid_count = valid.sum(dim=(1, 2)).clamp_min(1)
            boundary_diagnostics = {
                "semantic_boundary_fraction": boundary.float().mean(dim=-1),
                "semantic_boundary_shift": (
                    semantic_assign - assign
                ).abs().sum(dim=-1).mean(dim=-1),
                "semantic_boundary_tv": (edge_delta * valid).sum(dim=(1, 2))
                / valid_count,
            }

        amplitude_logits = self.region_amplitude(semantic_tokens).squeeze(-1)
        region_existence_logits = self.region_existence(semantic_tokens).squeeze(-1)
        if self.region_radius is None:
            replay_dropout_rng(
                semantic_tokens,
                (*semantic_tokens.shape[:-1], int(semantic_tokens.shape[-1]) // 2),
                probability=float(self.canonical_replay_dropout_probability),
                training=self.training,
            )
            radius = None
        else:
            radius = self.radius_min + (
                self.radius_max - self.radius_min
            ) * pt.sigmoid(self.region_radius(semantic_tokens)).squeeze(-1)

        valid_rs = (rs_idx >= 0) & (rs_idx < max_residues)
        safe_rs_idx = rs_idx.clamp(0, max(max_residues - 1, 0))
        surface_to_residue = (
            pt.nn.functional.one_hot(safe_rs_idx, num_classes=max_residues)
            .to(dtype=assign.dtype)
            .mul((rs_weight * valid_rs).unsqueeze(-1))
            .sum(dim=2)
        )
        region_to_residue = pt.bmm(
            semantic_assign.transpose(1, 2), surface_to_residue
        )
        region_to_residue = region_to_residue / region_to_residue.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        query = self.patch_queries
        query_count = int(query.shape[0])
        scale = math.sqrt(max(int(query.shape[-1]), 1))
        region_projected = self.region_proj(semantic_tokens)
        score = pt.einsum(
            "qd,bpd->bqp", self.query_proj(query), region_projected
        ) / scale
        seed_soft = pt.softmax(score, dim=-1)
        seed_hard = pt.nn.functional.one_hot(
            seed_soft.argmax(dim=-1), num_classes=patch_count
        ).to(dtype=seed_soft.dtype)
        seed = seed_hard + seed_soft - seed_soft.detach()
        compatibility = pt.sigmoid(score)
        adjacency_float = adjacency.to(device=device, dtype=score.dtype).clone()
        diagonal = pt.arange(patch_count, device=device)
        adjacency_float[:, diagonal, diagonal] = 1.0
        transition = adjacency_float / adjacency_float.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        membership = seed
        for _ in range(self.graph_hops):
            frontier = pt.bmm(membership, transition) * compatibility
            membership = 1.0 - (1.0 - membership) * (
                1.0 - frontier.clamp(0.0, 1.0)
            )
        region_activity = pt.sigmoid(amplitude_logits) * pt.sigmoid(
            region_existence_logits
        )
        functional_membership = membership * region_activity.unsqueeze(1)
        query_context = pt.bmm(membership, semantic_tokens) / membership.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        query_state = self.query_norm(
            query.view(1, query_count, -1) + query_context
        )
        existence_logits = self.patch_existence(query_state).squeeze(-1)
        surface_masks = pt.bmm(
            functional_membership, semantic_assign.transpose(1, 2)
        ).clamp(1e-4, 1.0 - 1e-4)
        residue_masks = pt.bmm(surface_masks, surface_to_residue)
        residue_support = surface_to_residue.sum(dim=1, keepdim=True).clamp_min(1e-8)
        residue_masks = (residue_masks / residue_support).clamp(1e-4, 1.0 - 1e-4)
        active_probability = pt.sigmoid(existence_logits).unsqueeze(-1) * residue_masks
        residual_probability = 1.0 - pt.exp(
            pt.log1p(-active_probability.clamp(max=1.0 - 1e-5)).sum(dim=1)
        )

        beta = self.beta_max * pt.tanh(self.residual_scale)
        centered_logit = pt.logit(
            residual_probability.clamp(1e-4, 1.0 - 1e-4)
        )
        residue_mask = (
            pt.arange(max_residues, device=device).unsqueeze(0)
            < pt.as_tensor(residue_counts, device=device).unsqueeze(1)
        )
        residue_count_tensor = pt.as_tensor(
            residue_counts, device=device, dtype=centered_logit.dtype
        ).clamp_min(1.0)
        logit_mean = (centered_logit * residue_mask).sum(dim=-1) / residue_count_tensor
        residual = beta * (centered_logit - logit_mean.unsqueeze(-1)).to(
            dtype=h_res.dtype
        )

        outputs = []
        region_weights = membership / membership.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        seed_region = seed_hard.argmax(dim=-1)
        for sample_index, residue_count in enumerate(residue_counts):
            output = {
                "region_amplitude_logits": amplitude_logits[sample_index],
                "region_existence_logits": region_existence_logits[sample_index],
                "region_to_residue": region_to_residue[
                    sample_index, :, :residue_count
                ],
                "residue_centers": centers[sample_index, :residue_count].detach(),
                "adjacency": adjacency[sample_index].detach(),
                "semantic_assignment": semantic_assign[sample_index],
                "set_queries": query_state[sample_index],
                "set_region_weights": region_weights[sample_index],
                "set_surface_masks": surface_masks[sample_index],
                "set_residue_masks": residue_masks[sample_index, :, :residue_count],
                "set_existence_logits": existence_logits[sample_index],
                "set_seed_region": seed_region[sample_index],
                "semantic_residual_beta": beta,
                "semantic_residual_probability": residual_probability[
                    sample_index, :residue_count
                ],
                "residual": residual[sample_index, :residue_count],
            }
            if radius is not None:
                output["region_radius"] = radius[sample_index]
            output.update(
                {
                    key: value[sample_index]
                    for key, value in boundary_diagnostics.items()
                }
            )
            outputs.append(output)
        return outputs

    def forward(
        self,
        h_res,
        h_surf,
        centers,
        surface_pos,
        surface_normal,
        surface_graph_idx,
        patch_tokens,
        patch_pos,
        assign,
        adjacency,
        rs_idx,
        rs_weight,
    ):
        if not self.enabled:
            return {"residual": h_res.new_zeros((h_res.shape[0],))}
        semantic_tokens = patch_tokens.detach() if self.detach_backbone else patch_tokens
        semantic_surface = h_surf.detach() if self.detach_backbone else h_surf
        semantic_assign = assign
        boundary_diagnostics = {}
        if self.mode == "graph_motif_refine":
            semantic_assign, boundary_diagnostics = self._local_boundary_refinement(
                assign, semantic_surface, semantic_tokens, surface_graph_idx
            )

        amplitude_logits = self.region_amplitude(semantic_tokens).squeeze(-1)
        region_existence_logits = self.region_existence(semantic_tokens).squeeze(-1)
        if self.region_radius is None:
            replay_dropout_rng(
                semantic_tokens,
                (int(semantic_tokens.shape[0]), int(semantic_tokens.shape[-1]) // 2),
                probability=float(self.canonical_replay_dropout_probability),
                training=self.training,
            )
            radius = None
        else:
            radius = self.radius_min + (self.radius_max - self.radius_min) * pt.sigmoid(
                self.region_radius(semantic_tokens)
            ).squeeze(-1)
        surface_to_residue = self._surface_to_residue_matrix(
            semantic_assign, rs_idx, rs_weight, int(h_res.shape[0])
        )
        region_to_residue = self._region_to_residue(semantic_assign, surface_to_residue)
        output = {
            "region_amplitude_logits": amplitude_logits,
            "region_existence_logits": region_existence_logits,
            "region_to_residue": region_to_residue,
            "residue_centers": centers.detach(),
            "adjacency": adjacency.detach(),
            "semantic_assignment": semantic_assign,
            **boundary_diagnostics,
        }
        if radius is not None:
            output["region_radius"] = radius

        residual_probability = None
        if self.mode in {"gaussian", "sparse_noisyor"}:
            geodesic_distance, center_index = self._graph_geodesic_distances(
                surface_pos, surface_normal, surface_graph_idx, patch_pos
            )
            output["patch_geodesic_distances"] = geodesic_distance.detach()
            output["patch_center_surface_index"] = center_index.detach()
        if self.mode in {"gaussian", "sparse_noisyor"}:
            residual_probability, active_mask, diagnostics = self._sparse_region_noisy_or(
                centers,
                surface_pos,
                geodesic_distance,
                amplitude_logits,
                region_existence_logits,
                radius,
            )
            output["active_region_mask"] = active_mask
            output.update(diagnostics)
        elif self.mode in {"graph_motif", "graph_motif_refine", "setpatch", "full"}:
            motif_output = self._graph_constrained_motifs(
                semantic_tokens,
                semantic_assign,
                adjacency,
                surface_to_residue,
                amplitude_logits,
                region_existence_logits,
            )
            output.update({f"set_{key}": value for key, value in motif_output.items() if key != "probability"})
            residual_probability = motif_output["probability"]

        if self.mode in {"sparse_probe", "amplitude"} or residual_probability is None:
            residual = h_res.new_zeros((h_res.shape[0],))
        else:
            beta = self.beta_max * pt.tanh(self.residual_scale)
            residual = beta * self._centered_logit(residual_probability).to(dtype=h_res.dtype)
            output["semantic_residual_beta"] = beta
            output["semantic_residual_probability"] = residual_probability
        output["residual"] = residual
        return output
