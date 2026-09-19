"""PeSTo with a plug-and-play M4 Gaussian surface-interaction block.

The block reuses the exact M4 surface-path module families (Gaussian residue
projection, pair-Gaussian correction, multigraph GVP surface encoder, connected
patch cluster, surface-to-residue attention, and three bidirectional co-update
rounds) and is inserted between PeSTo's residue pooling and its decoder.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch as pt

from .modules.cross_scale import (
    AttentionSurfaceToResiduePool,
)
from .modules.gaussian_projection import (
    build_gaussian_projector,
)
from .modules.pair_gaussian import (
    build_pair_gaussian_modules,
)
from .modules.residue_backbone import GlobalContextFeedback
from .modules.semantic_patch import SemanticConnectedPatchHead
from .modules.surface_encoder import (
    DynamicSurfacePatchCluster,
    SurfacePointGVPGraphEncoder,
)


class S1GaussianInteractionBlock(pt.nn.Module):
    """Faithful M4 surface interaction path with three co-update rounds.

    E00P execution semantics:
    - residue-to-surface projection happens only in round 0 (pair_gaussian_only)
    - pair-Gaussian correction runs in every round (fusion=deep)
    - surface multigraph uses k16 + k32 GVP encoders with a learned router
    - surface patch mode is connected_pool_adj
    - round-end residue-to-surface update is disabled (E00P skip mode)
    - the residue state is co-updated between rounds through the core gate
    """

    def __init__(
        self,
        input_dim: int = 32,
        gaussian_dim: int = 128,
        vector_dim: int = 16,
        dropout: float = 0.1,
        surface_residue_topk: int = 16,
        residue_surface_topk: int = 32,
        sigma_tangent: float = 6.0,
        sigma_normal: float = 3.0,
        multiscale_sigmas: str = "4:2,8:4,14:7",
        surface_graph_k: int = 16,
        surface_extra_graph_ks: tuple[int, ...] = (32,),
        surface_layers: int = 2,
        surface_attention_prior_weight: float = 0.25,
        surface_attention_sigma_tangent: float = 8.0,
        surface_attention_sigma_normal: float = 4.0,
        surface_patch_count: int = 32,
        surface_patch_beta: float = 0.1,
        surface_patch_attention_heads: int = 4,
        pair_gaussian_topk: int = 16,
        pair_gaussian_beta: float = 0.05,
        pair_gaussian_sigma_tangent: float = 8.0,
        pair_gaussian_sigma_normal: float = 4.0,
        co_update_rounds: int = 3,
        zero_init_output: bool = True,
        architecture: str = "base",
        ablation_no_surface_vector: bool = False,
        ablation_no_residue_feedback: bool = False,
    ):
        super().__init__()
        architecture = str(architecture)
        if architecture not in {"base", "vg", "sem"}:
            raise ValueError("architecture must be one of: base, vg, sem")
        self.architecture = architecture
        self.use_surface_vector = (
            architecture in {"vg", "sem"} and not ablation_no_surface_vector
        )
        self.use_global_feedback = architecture in {"vg", "sem"}
        self.use_residue_global_feedback = (
            self.use_global_feedback and not ablation_no_residue_feedback
        )
        self.use_surface_global_feedback = self.use_global_feedback
        self.use_semantic = architecture == "sem"
        self.input_dim = int(input_dim)
        self.gaussian_dim = int(gaussian_dim)
        self.vector_dim = int(vector_dim)
        self.surface_graph_k = int(surface_graph_k)
        self.surface_extra_graph_ks = tuple(int(k) for k in surface_extra_graph_ks)
        self.co_update_rounds = int(co_update_rounds)
        self.last_fused = None
        self.collect_round_trace = False
        self.last_round_trace: list[dict[str, Any]] = []

        use_adapter = self.input_dim != self.gaussian_dim
        self.input_adapter = (
            pt.nn.Linear(self.input_dim, self.gaussian_dim)
            if use_adapter
            else pt.nn.Identity()
        )
        self.output_projection = pt.nn.Linear(self.gaussian_dim, self.input_dim)

        self.res_to_surface = build_gaussian_projector(
            projection_mode="per_residue_sigma",
            dim=self.gaussian_dim,
            surface_residue_topk=surface_residue_topk,
            sigma_tangent=sigma_tangent,
            sigma_normal=sigma_normal,
            multiscale_sigmas=multiscale_sigmas,
            residue_amplitude_mode="none",
            residue_amplitude_min=0.25,
            residue_amplitude_max=2.0,
            phys_feature_dim=0,
            true_feature_dim=0,
            atom_feature_dim=0,
            surface_patch_count=surface_patch_count,
            dropout=dropout,
        )
        self.surface_encoder = SurfacePointGVPGraphEncoder(
            dim=self.gaussian_dim,
            vector_dim=vector_dim,
            geom_dim=6,
            dropout=dropout,
            layers=surface_layers,
        )
        self.surface_extra_encoders = pt.nn.ModuleList(
            SurfacePointGVPGraphEncoder(
                dim=self.gaussian_dim,
                vector_dim=vector_dim,
                geom_dim=6,
                dropout=dropout,
                layers=surface_layers,
            )
            for _k in self.surface_extra_graph_ks
        )
        self.surface_graph_logits = pt.nn.Parameter(
            pt.zeros(1 + len(self.surface_extra_graph_ks))
        )
        self.surface_graph_fusion_norm = pt.nn.LayerNorm(self.gaussian_dim)
        self.surface_to_residue = AttentionSurfaceToResiduePool(
            dim=self.gaussian_dim,
            surface_topk=residue_surface_topk,
            dropout=dropout,
            mode="attention",
            gaussian_prior_weight=surface_attention_prior_weight,
            sigma_tangent=surface_attention_sigma_tangent,
            sigma_normal=surface_attention_sigma_normal,
            phys_feature_dim=0,
            phys_attention_bias_weight=0.0,
            phys_context_weight=0.0,
            phys_message_weight=0.0,
            surface_vector_bridge_mode=(
                "invariant" if self.use_surface_vector else "none"
            ),
            surface_vector_dim=vector_dim,
        )
        if self.use_global_feedback:
            self.residue_global_feedback = GlobalContextFeedback(
                dim=self.gaussian_dim, dropout=dropout
            )
            self.surface_global_feedback = GlobalContextFeedback(
                dim=self.gaussian_dim, dropout=dropout
            )
        if self.use_semantic:
            self.semantic_patch_head = SemanticConnectedPatchHead(
                dim=self.gaussian_dim,
                query_count=8,
                mode="graph_motif_refine",
                beta=0.03,
                radius_min=2.0,
                radius_max=16.0,
                adjacency_weight=0.25,
                dropout=dropout,
                attention_heads=surface_patch_attention_heads,
                active_topk=4,
                graph_hops=2,
                geodesic_max_steps=96,
                geodesic_normal_penalty=1.0,
                geodesic_tangent_penalty=1.0,
                detach_backbone=True,
                boundary_mix=0.1,
                boundary_temperature=0.7,
            )
            self.semantic_patch_head.set_boundary_refinement_scale(0.0)
        self.last_semantic_outputs: list[dict[str, pt.Tensor]] = []
        self.surface_patch_cluster = DynamicSurfacePatchCluster(
            dim=self.gaussian_dim,
            num_patches=surface_patch_count,
            mode="connected_pool_adj",
            beta=surface_patch_beta,
            dropout=dropout,
            attention_heads=surface_patch_attention_heads,
        )

        gaussian_modules = build_pair_gaussian_modules(
            dim=self.gaussian_dim,
            vector_dim=vector_dim,
            graph_views=("seq", "k16", "k32", "exposure"),
            dropout=dropout,
            pair_mode="k_nearest",
            pair_topk=pair_gaussian_topk,
            pair_beta=pair_gaussian_beta,
            pair_sigma_tangent=pair_gaussian_sigma_tangent,
            pair_sigma_normal=pair_gaussian_sigma_normal,
            pair_sigma_mode="fixed",
            pair_sigma_min=0.5,
            pair_joint_mode="joint_only",
            pair_fusion_mode="deep",
            pair_content_mode="none",
            pair_content_weight=1.0,
            pair_rho_condition_mode="none",
            pair_vector_condition_mode="none",
            pair_single_compete_mode="none",
            pair_single_rescue_beta=0.25,
            pair_beta_mode="fixed",
            pair_beta_min=0.0,
            pair_beta_max=0.3,
            pair_beta_init=0.05,
            pair_multiview_mode="none",
            pair_multiview_beta=0.1,
            surface_co_update_mode="bidir",
            surface_co_update_layers=self.co_update_rounds,
            core_cycles=1,
            triplet_mode="none",
            triplet_topk=8,
            triplet_beta=0.0,
            triplet_sigma_tangent=8.0,
            triplet_sigma_normal=4.0,
            triplet_score_temperature=1.0,
            triplet_score_clamp=0.0,
            triplet_preselect_topm=0,
            triplet_beta_mode="fixed",
            triplet_beta_min=0.0,
            triplet_beta_max=0.2,
            triplet_beta_init=0.03,
            global_feedback_rounds=2,
            surface_global_feedback_rounds=2,
        )
        self.pair_gaussian = gaussian_modules["pair"]
        self.pair_gaussian_deep_beta_logits = gaussian_modules["pair_deep_beta"]

        self.same_level_fusion = pt.nn.Sequential(
            pt.nn.LayerNorm(self.gaussian_dim * 2),
            pt.nn.Linear(self.gaussian_dim * 2, self.gaussian_dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(self.gaussian_dim, self.gaussian_dim),
        )
        self.core_residue_update = pt.nn.Sequential(
            pt.nn.LayerNorm(self.gaussian_dim * 2),
            pt.nn.Linear(self.gaussian_dim * 2, self.gaussian_dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(self.gaussian_dim, self.gaussian_dim),
        )
        self.core_residue_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(self.gaussian_dim * 2),
            pt.nn.Linear(self.gaussian_dim * 2, self.gaussian_dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(self.gaussian_dim, self.gaussian_dim),
            pt.nn.Sigmoid(),
        )
        self.core_norm = pt.nn.LayerNorm(self.gaussian_dim)
        self.residue_input_norm = pt.nn.LayerNorm(self.gaussian_dim)
        if zero_init_output:
            pt.nn.init.zeros_(self.output_projection.weight)
            pt.nn.init.zeros_(self.output_projection.bias)

    def gaussian_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_semantic_state(self) -> None:
        self.last_semantic_outputs = []

    def set_semantic_epoch(self, epoch: int) -> None:
        if not self.use_semantic:
            return
        active = int(epoch) + 1 >= 3
        self.semantic_patch_head.set_boundary_refinement_scale(
            1.0 if active else 0.0
        )

    def _surface_graph_for_k(
        self,
        cache: Mapping[str, Any],
        surface_pos: pt.Tensor,
        graph_k: int,
    ) -> tuple[pt.Tensor, pt.Tensor]:
        graph_k = int(graph_k)
        if graph_k <= 0:
            return cache.get("surface_graph_idx"), cache.get("surface_graph_dist")
        cached_idx = cache.get(f"surface_graph_idx_k{graph_k}")
        cached_dist = cache.get(f"surface_graph_dist_k{graph_k}")
        if cached_idx is not None and cached_dist is not None and cached_idx.numel() > 0:
            return cached_idx, cached_dist
        if graph_k == self.surface_graph_k:
            graph_idx = cache.get("surface_graph_idx")
            graph_dist = cache.get("surface_graph_dist")
            if graph_idx is not None and graph_idx.numel() > 0:
                return graph_idx, graph_dist
        idx_name = f"surface_graph_idx_k{graph_k}"
        dist_name = f"surface_graph_dist_k{graph_k}"
        points = int(surface_pos.shape[0])
        if points <= 1:
            graph_idx = surface_pos.new_empty((points, 0), dtype=pt.long)
            graph_dist = surface_pos.new_empty((points, 0))
            cache[idx_name] = graph_idx
            cache[dist_name] = graph_dist
            return graph_idx, graph_dist
        k = min(graph_k + 1, points)
        dist = pt.cdist(surface_pos, surface_pos).clamp_min(0.0)
        graph_dist, graph_idx = pt.topk(dist, k=k, dim=-1, largest=False)
        graph_idx = graph_idx[:, 1:].contiguous()
        graph_dist = graph_dist[:, 1:].contiguous()
        cache[idx_name] = graph_idx
        cache[dist_name] = graph_dist
        return graph_idx, graph_dist

    def _encode_surface_multigraph(
        self,
        h_surf_state: pt.Tensor,
        surface_geom: pt.Tensor,
        surface_pos: pt.Tensor,
        surface_normal: pt.Tensor,
        cache: Mapping[str, Any],
    ) -> tuple[pt.Tensor, pt.Tensor | None]:
        encoders = [self.surface_encoder, *list(self.surface_extra_encoders)]
        graph_ks = [self.surface_graph_k, *list(self.surface_extra_graph_ks)]
        outputs = []
        vectors = []
        for encoder, graph_k in zip(encoders, graph_ks):
            graph_idx, graph_dist = self._surface_graph_for_k(cache, surface_pos, graph_k)
            outputs.append(
                encoder(
                    h_surf_state,
                    surface_geom,
                    surface_pos,
                    surface_normal,
                    graph_idx,
                    graph_dist=graph_dist,
                    initial_vectors=None,
                )
            )
            if self.use_surface_vector:
                surface_vectors = getattr(encoder, "last_surface_vectors", None)
                if surface_vectors is not None:
                    vectors.append(surface_vectors)
        if len(outputs) == 1:
            fused_vectors = vectors[0] if len(vectors) == len(outputs) else None
            return outputs[0], fused_vectors
        weights = pt.softmax(
            self.surface_graph_logits[: len(outputs)], dim=0
        ).to(device=h_surf_state.device, dtype=h_surf_state.dtype)
        stacked = pt.stack(outputs, dim=0)
        fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
        fused_vectors = None
        if self.use_surface_vector and len(vectors) == len(outputs):
            vector_stack = pt.stack(vectors, dim=0)
            fused_vectors = (
                weights.view(-1, 1, 1, 1) * vector_stack
            ).sum(dim=0)
        return self.surface_graph_fusion_norm(fused), fused_vectors

    def forward(
        self,
        residue_state: pt.Tensor,
        residue_vectors: pt.Tensor | None,
        residue_centers: pt.Tensor,
        surface_cache: Mapping[str, Any],
    ) -> pt.Tensor:
        self.reset_semantic_state()
        self.last_round_trace = []
        h_res = self.residue_input_norm(self.input_adapter(residue_state))
        if self.use_residue_global_feedback:
            residue_mask = h_res.new_ones((1, h_res.shape[0]))
            h_res_global = h_res.unsqueeze(0)
            for _ in range(2):
                h_res_global = self.residue_global_feedback(h_res_global, residue_mask)
            h_res = h_res_global.squeeze(0)
        h_surf_state: pt.Tensor | None = None
        surface_pos = surface_normal = surface_geom = rs_idx = rs_weight = None
        surf_context: pt.Tensor | None = None
        connected_partition = None

        for core_idx in range(self.co_update_rounds):
            round_trace: dict[str, Any] = {}
            if self.collect_round_trace:
                round_trace["core_round"] = int(core_idx)
                round_trace["residue_state_pre"] = h_res.detach().to(device="cpu", dtype=pt.float32).contiguous()
            if core_idx == 0:
                (
                    h_surf_state,
                    surface_pos,
                    surface_normal,
                    surface_geom,
                    rs_idx,
                    rs_weight,
                ) = self.res_to_surface(h_res, residue_centers, surface_cache)
            assert h_surf_state is not None and surface_pos is not None
            assert surface_normal is not None and surface_geom is not None
            assert rs_idx is not None and rs_weight is not None

            if self.pair_gaussian is not None:
                beta_override = None
                if self.pair_gaussian_deep_beta_logits is not None and core_idx > 0:
                    beta_idx = min(
                        core_idx - 1,
                        int(self.pair_gaussian_deep_beta_logits.numel()) - 1,
                    )
                    beta_logit = self.pair_gaussian_deep_beta_logits[beta_idx].to(
                        device=h_res.device, dtype=h_res.dtype
                    )
                    beta_override = pt.sigmoid(beta_logit)
                h_surf_state, _diag = self.pair_gaussian(
                    h_res,
                    residue_centers,
                    surface_pos,
                    surface_normal,
                    surface_geom,
                    h_surf_state,
                    beta_override=beta_override,
                )
                if self.collect_round_trace:
                    round_trace["route_records"] = [
                        dict(record)
                        for record in (getattr(self.pair_gaussian, "last_top_records", []) or [])
                    ]

            h_surf, surface_vectors = self._encode_surface_multigraph(
                h_surf_state,
                surface_geom,
                surface_pos,
                surface_normal,
                surface_cache,
            )
            if self.use_surface_global_feedback:
                surface_mask = h_surf.new_ones((1, h_surf.shape[0]))
                h_surf_global = h_surf.unsqueeze(0)
                for _ in range(2):
                    h_surf_global = self.surface_global_feedback(
                        h_surf_global, surface_mask
                    )
                h_surf = h_surf_global.squeeze(0)
            surf_context, _attention = self.surface_to_residue(
                h_res=h_res,
                h_surf=h_surf,
                residue_centers=residue_centers,
                surface_pos=surface_pos,
                surface_normal=surface_normal,
                residue_phys_features=None,
                surface_phys_features=None,
                surface_vectors=surface_vectors if self.use_surface_vector else None,
                residue_vectors=residue_vectors if self.use_surface_vector else None,
                geometry_cache=None,
            )
            if self.collect_round_trace:
                round_trace["surface_context_pre_patch"] = surf_context.detach().to(device="cpu", dtype=pt.float32).contiguous()
                round_trace["residue_state_before_core_update"] = h_res.detach().to(device="cpu", dtype=pt.float32).contiguous()
            if self.surface_patch_cluster is not None:
                if connected_partition is None:
                    connected_graph_idx, _dist = self._surface_graph_for_k(
                        surface_cache, surface_pos, self.surface_graph_k
                    )
                    connected_partition = self.surface_patch_cluster.connected_partition(
                        surface_pos, connected_graph_idx
                    )
                h_surf, surf_context, _patch_diag = self.surface_patch_cluster(
                    h_res,
                    h_surf,
                    surface_pos,
                    surface_normal,
                    surface_geom,
                    rs_idx,
                    rs_weight,
                    surf_context=surf_context,
                    patch_state=None,
                    surface_lappe=None,
                    connected_partition=connected_partition,
                )
                final_patch_diag = _patch_diag
            if self.collect_round_trace:
                round_trace["surface_context_post_patch"] = surf_context.detach().to(device="cpu", dtype=pt.float32).contiguous()

            if core_idx + 1 < self.co_update_rounds:
                core_input = pt.cat([h_res, surf_context], dim=-1)
                h_res = self.core_norm(
                    h_res
                    + self.core_residue_gate(core_input) * self.core_residue_update(core_input)
                )
                h_surf_state = h_surf
            if self.collect_round_trace:
                round_trace["residue_state_post"] = h_res.detach().to(device="cpu", dtype=pt.float32).contiguous()
                round_trace["residue_delta"] = (
                    round_trace["residue_state_post"] - round_trace["residue_state_pre"]
                ).contiguous()
                self.last_round_trace.append(round_trace)

        assert surf_context is not None
        fused = self.same_level_fusion(pt.cat([h_res, surf_context], dim=-1))
        self.last_fused = fused
        update = self.output_projection(fused)
        if self.use_semantic:
            required_patch_keys = {
                "patch_tokens",
                "patch_positions",
                "patch_assignment",
                "patch_adjacency",
                "patch_surface_graph_idx",
            }
            missing_patch_keys = required_patch_keys.difference(final_patch_diag)
            if missing_patch_keys:
                raise RuntimeError(
                    "semantic patch head missing connected patch outputs: "
                    + ", ".join(sorted(missing_patch_keys))
                )
            semantic_output = self.semantic_patch_head(
                h_res=h_res,
                h_surf=h_surf,
                centers=residue_centers,
                surface_pos=surface_pos,
                surface_normal=surface_normal,
                surface_graph_idx=final_patch_diag["patch_surface_graph_idx"],
                patch_tokens=final_patch_diag["patch_tokens"],
                patch_pos=final_patch_diag["patch_positions"],
                assign=final_patch_diag["patch_assignment"],
                adjacency=final_patch_diag["patch_adjacency"],
                rs_idx=rs_idx,
                rs_weight=rs_weight,
            )
            self.last_semantic_outputs.append(semantic_output)
        return residue_state + update


class ParameterMatchedResidueMLP(pt.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.network = pt.nn.Sequential(
            pt.nn.LayerNorm(input_dim),
            pt.nn.Linear(input_dim, hidden_dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            *[
                module
                for _ in range(5)
                for module in (
                    pt.nn.Linear(hidden_dim, hidden_dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                )
            ],
            pt.nn.Linear(hidden_dim, output_dim),
            pt.nn.GELU(),
        )

    def forward(self, residue_embedding: pt.Tensor) -> pt.Tensor:
        return self.network(residue_embedding)


class PeSToGaussianModel(pt.nn.Module):
    """Official PeSTo followed by the M4 Gaussian interaction block."""

    def __init__(
        self,
        pesto_config: Mapping[str, Any],
        gaussian_dim: int,
        co_update_rounds: int = 3,
        dropout: float = 0.1,
        zero_init_output: bool = True,
        architecture: str = "base",
        direct_protein_residual: bool = False,
        direct_protein_head: bool = False,
        bypass_gaussian: bool = False,
        parameter_matched_mlp: bool = False,
        ablation_no_surface_vector: bool = False,
        ablation_no_residue_feedback: bool = False,
    ):
        super().__init__()
        from .pesto.model import Model as PeSToModel

        self.pesto = PeSToModel(pesto_config)
        residue_dim = int(pesto_config["spl"]["N1"])
        self.direct_protein_residual = bool(direct_protein_residual)
        self.direct_protein_head = bool(direct_protein_head)
        self.bypass_gaussian = bool(bypass_gaussian)
        self.parameter_matched_mlp = bool(parameter_matched_mlp)
        self.collect_round_trace = False
        self.last_batch_round_traces: list[list[dict[str, Any]]] = []
        if self.direct_protein_residual and self.direct_protein_head:
            raise ValueError(
                "direct-protein-residual and direct-protein-head are mutually exclusive"
            )
        if self.bypass_gaussian and not self.direct_protein_head:
            raise ValueError("bypass-gaussian requires direct-protein-head")
        if self.parameter_matched_mlp and not (self.bypass_gaussian and self.direct_protein_head):
            raise ValueError("parameter-matched-mlp requires bypass-gaussian and direct-protein-head")
        decoder_dim = int(pesto_config["dm"]["N1"])
        if self.bypass_gaussian:
            self.gaussian = None
            if self.parameter_matched_mlp:
                self.decoder_projection = ParameterMatchedResidueMLP(
                    input_dim=decoder_dim,
                    hidden_dim=591,
                    output_dim=int(gaussian_dim),
                    dropout=dropout,
                )
            else:
                self.decoder_projection = pt.nn.Linear(decoder_dim, int(gaussian_dim))
        else:
            self.gaussian = S1GaussianInteractionBlock(
                input_dim=decoder_dim if self.direct_protein_head else residue_dim,
                gaussian_dim=int(gaussian_dim),
                vector_dim=16,
                dropout=dropout,
                co_update_rounds=int(co_update_rounds),
                zero_init_output=zero_init_output,
                architecture=architecture,
                ablation_no_surface_vector=ablation_no_surface_vector,
                ablation_no_residue_feedback=ablation_no_residue_feedback,
            )
        if self.direct_protein_residual:
            self.protein_residual_head = pt.nn.Sequential(
                pt.nn.LayerNorm(int(gaussian_dim) + residue_dim * 3),
                pt.nn.Linear(int(gaussian_dim) + residue_dim * 3, int(gaussian_dim)),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(int(gaussian_dim), 1),
            )
            pt.nn.init.xavier_uniform_(self.protein_residual_head[-1].weight)
            pt.nn.init.zeros_(self.protein_residual_head[-1].bias)
            self.protein_residual_scale = pt.nn.Parameter(pt.zeros((), dtype=pt.float32))
        if self.direct_protein_head:
            self.protein_output_head = pt.nn.Sequential(
                pt.nn.LayerNorm(int(gaussian_dim)),
                pt.nn.Linear(int(gaussian_dim), int(gaussian_dim)),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(int(gaussian_dim), int(gaussian_dim)),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(int(gaussian_dim), 5),
            )
            for linear in self.protein_output_head:
                if isinstance(linear, pt.nn.Linear):
                    pt.nn.init.xavier_uniform_(linear.weight)
                    pt.nn.init.zeros_(linear.bias)

    def pesto_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.pesto.parameters())

    def gaussian_parameter_count(self) -> int:
        if self.bypass_gaussian:
            count = sum(parameter.numel() for parameter in self.decoder_projection.parameters())
        else:
            count = self.gaussian.gaussian_parameter_count()
        if self.direct_protein_residual:
            count += sum(parameter.numel() for parameter in self.protein_residual_head.parameters())
            count += self.protein_residual_scale.numel()
        if self.direct_protein_head:
            count += sum(parameter.numel() for parameter in self.protein_output_head.parameters())
        return count

    def _decode_with_semantic_residual(
        self,
        decoded: pt.Tensor,
        gaussian_update: pt.Tensor | None,
        gaussian_feature: pt.Tensor | None,
    ) -> pt.Tensor:
        logits = self.pesto.dm(decoded)
        if not self.gaussian.use_semantic:
            return logits
        if not self.gaussian.last_semantic_outputs:
            raise RuntimeError("semantic forward did not produce sample outputs")
        residual = pt.cat(
            [output["residual"] for output in self.gaussian.last_semantic_outputs],
            dim=0,
        )
        return logits + residual.unsqueeze(-1)

    def _direct_protein_residual(
        self,
        decoded: pt.Tensor,
        gaussian_update: pt.Tensor,
        gaussian_feature: pt.Tensor,
    ) -> pt.Tensor:
        residual_input = pt.cat(
            [gaussian_feature, decoded, gaussian_update], dim=1
        )
        return self.protein_residual_head(residual_input).squeeze(-1)

    def _direct_protein_output(
        self,
        gaussian_feature: pt.Tensor,
    ) -> pt.Tensor:
        return self.protein_output_head(gaussian_feature)

    def pesto_residue_features(
        self,
        X: pt.Tensor,
        ids_topk: pt.Tensor,
        q0: pt.Tensor,
        M: pt.Tensor,
    ) -> tuple[pt.Tensor, pt.Tensor]:
        encoded = self.pesto.em(q0)
        initial_state = pt.zeros(
            (encoded.shape[0] + 1, X.shape[1], encoded.shape[1]), device=X.device
        )
        from .pesto.operations import unpack_state_features

        packed_q, packed_ids, packed_dist, packed_r = unpack_state_features(
            X, ids_topk, encoded
        )
        updated_q, updated_p, _, _, _ = self.pesto.sum(
            (packed_q, initial_state, packed_ids, packed_dist, packed_r)
        )
        return self.pesto.spl(updated_q[1:], updated_p[1:], M)

    def _residue_centers(self, X: pt.Tensor, M: pt.Tensor) -> pt.Tensor:
        weights = M.to(dtype=X.dtype)
        denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        return weights.transpose(0, 1).matmul(X) / denom

    def _gaussian_forward(self, *args: Any, **kwargs: Any) -> pt.Tensor:
        if self.gaussian is None:
            raise RuntimeError("Gaussian forward requested for a bypassed model.")
        self.gaussian.collect_round_trace = bool(self.collect_round_trace)
        result = self.gaussian(*args, **kwargs)
        if self.collect_round_trace:
            self.last_batch_round_traces.append(self.gaussian.last_round_trace)
        return result

    def forward(
        self,
        X: pt.Tensor,
        ids_topk: pt.Tensor,
        q0: pt.Tensor,
        M: pt.Tensor,
        residue_centers: pt.Tensor | None = None,
        surface_cache: Mapping[str, Any] | None = None,
        atom_counts: pt.Tensor | tuple[int, ...] | list[int] | None = None,
        residue_counts: pt.Tensor | tuple[int, ...] | list[int] | None = None,
        frozen_residue_q: pt.Tensor | list[pt.Tensor] | None = None,
        frozen_residue_p: pt.Tensor | list[pt.Tensor] | None = None,
    ) -> pt.Tensor:
        self.last_batch_round_traces = []
        if not self.bypass_gaussian:
            self.gaussian.reset_semantic_state()
        if frozen_residue_q is None or frozen_residue_p is None:
            residue_q, residue_p = self.pesto_residue_features(X, ids_topk, q0, M)
        elif isinstance(frozen_residue_q, list):
            if not isinstance(frozen_residue_p, list):
                raise ValueError("frozen_residue_p must be a list when residue_q is a list")
            if len(frozen_residue_q) != len(frozen_residue_p):
                raise ValueError("frozen residue feature batch lengths do not match")
            residue_q = pt.cat(frozen_residue_q, dim=0)
            residue_p = pt.cat(frozen_residue_p, dim=0)
        else:
            residue_q = frozen_residue_q
            residue_p = frozen_residue_p
        original_residue_q = residue_q
        if self.bypass_gaussian:
            decoded = pt.cat([residue_q, pt.norm(residue_p, dim=1)], dim=1)
            residue_embedding = self.pesto.dm[:-1](decoded)
            decoder_feature = self.decoder_projection(residue_embedding)
            return self._direct_protein_output(decoder_feature)
        if self.direct_protein_head:
            decoded = pt.cat([residue_q, pt.norm(residue_p, dim=1)], dim=1)
            residue_embedding = self.pesto.dm[:-1](decoded)
            if frozen_residue_q is not None and isinstance(frozen_residue_q, list):
                if not isinstance(surface_cache, list):
                    raise ValueError("Frozen residue feature batches require surface_cache batches")
                if residue_centers is None or residue_counts is None:
                    raise ValueError("Frozen residue feature batches require centers and residue_counts")
                if not isinstance(residue_centers, list):
                    raise ValueError("Frozen residue feature centers must be a list")
                residue_counts = [int(value) for value in residue_counts]
                if len(surface_cache) != len(residue_counts) or len(residue_centers) != len(residue_counts):
                    raise ValueError("Frozen residue feature batch count mismatch")
                residue_start = 0
                gaussian_features = []
                for sample_index, current_surface in enumerate(surface_cache):
                    residue_end = residue_start + residue_counts[sample_index]
                    self._gaussian_forward(
                        residue_embedding[residue_start:residue_end],
                        residue_p[:, residue_start:residue_end, :].transpose(0, 1).contiguous(),
                        residue_centers[sample_index],
                        current_surface,
                    )
                    gaussian_features.append(self.gaussian.last_fused)
                    residue_start = residue_end
                return self._direct_protein_output(pt.cat(gaussian_features, dim=0))

            batched_surface = isinstance(surface_cache, list)
            if surface_cache is None:
                raise ValueError("PeSToGaussianModel requires a compact surface cache.")
            if not batched_surface:
                if residue_centers is None:
                    residue_centers = self._residue_centers(X, M)
                self._gaussian_forward(
                    residue_embedding,
                    residue_p.transpose(0, 1).contiguous(),
                    residue_centers,
                    surface_cache,
                )
                return self._direct_protein_output(self.gaussian.last_fused)

            if residue_counts is None or atom_counts is None:
                raise ValueError("Batched surface caches require atom_counts and residue_counts.")
            atom_counts = [int(value) for value in atom_counts]
            residue_counts = [int(value) for value in residue_counts]
            if sum(atom_counts) != int(X.shape[0]):
                raise ValueError("Batched atom_counts do not match packed coordinates.")
            if sum(residue_counts) != int(residue_embedding.shape[0]):
                raise ValueError("Batched residue_counts do not match decoder residues.")
            if len(surface_cache) != len(atom_counts) or len(surface_cache) != len(residue_counts):
                raise ValueError("Batched surface cache count mismatch.")
            atom_start = residue_start = 0
            gaussian_features = []
            for sample_index, current_surface in enumerate(surface_cache):
                atom_end = atom_start + atom_counts[sample_index]
                residue_end = residue_start + residue_counts[sample_index]
                current_centers = self._residue_centers(
                    X[atom_start:atom_end], M[atom_start:atom_end, residue_start:residue_end]
                )
                self._gaussian_forward(
                    residue_embedding[residue_start:residue_end],
                    residue_p[:, residue_start:residue_end, :].transpose(0, 1).contiguous(),
                    current_centers,
                    current_surface,
                )
                gaussian_features.append(self.gaussian.last_fused)
                atom_start = atom_end
                residue_start = residue_end
            return self._direct_protein_output(pt.cat(gaussian_features, dim=0))

        batched_surface = isinstance(surface_cache, list)
        if frozen_residue_q is not None and isinstance(frozen_residue_q, list):
            if not batched_surface:
                raise ValueError("Frozen residue feature batches require surface_cache batches")
            if residue_centers is None or residue_counts is None:
                raise ValueError("Frozen residue feature batches require centers and residue_counts")
            if not isinstance(residue_centers, list):
                raise ValueError("Frozen residue feature centers must be a list")
            residue_counts = [int(value) for value in residue_counts]
            if len(surface_cache) != len(residue_counts) or len(residue_centers) != len(residue_counts):
                raise ValueError("Frozen residue feature batch count mismatch")
            residue_start = 0
            updated_residue_states = []
            gaussian_features = []
            for sample_index, current_surface in enumerate(surface_cache):
                residue_end = residue_start + residue_counts[sample_index]
                updated_residue_states.append(
                    self._gaussian_forward(
                        residue_q[residue_start:residue_end],
                        residue_p[:, residue_start:residue_end, :].transpose(0, 1).contiguous(),
                        residue_centers[sample_index],
                        current_surface,
                    )
                )
                gaussian_features.append(self.gaussian.last_fused)
                residue_start = residue_end
            decoded = pt.cat(
                [pt.cat(updated_residue_states, dim=0), pt.norm(residue_p, dim=1)], dim=1
            )
            gaussian_update = pt.cat(updated_residue_states, dim=0) - original_residue_q
            gaussian_feature = pt.cat(gaussian_features, dim=0)
            return self._decode_with_semantic_residual(
                decoded, gaussian_update, gaussian_feature
            )
        if surface_cache is None:
            raise ValueError("PeSToGaussianModel requires a compact surface cache.")
        if not batched_surface:
            if residue_centers is None:
                residue_centers = self._residue_centers(X, M)
            residue_q = self._gaussian_forward(
                residue_q,
                residue_p.transpose(0, 1).contiguous(),
                residue_centers,
                surface_cache,
            )
            gaussian_update = residue_q - original_residue_q
            gaussian_feature = self.gaussian.last_fused
        else:
            if residue_counts is None or atom_counts is None:
                raise ValueError("Batched surface caches require atom_counts and residue_counts.")
            atom_counts = [int(value) for value in atom_counts]
            residue_counts = [int(value) for value in residue_counts]
            if sum(atom_counts) != int(X.shape[0]):
                raise ValueError("Batched atom_counts do not match packed coordinates.")
            if sum(residue_counts) != int(residue_q.shape[0]):
                raise ValueError("Batched residue_counts do not match pooled residues.")
            if len(surface_cache) != len(atom_counts) or len(surface_cache) != len(residue_counts):
                raise ValueError("Batched surface cache count mismatch.")
            atom_start = residue_start = 0
            updated_residue_states = []
            gaussian_features = []
            for sample_index, current_surface in enumerate(surface_cache):
                atom_end = atom_start + atom_counts[sample_index]
                residue_end = residue_start + residue_counts[sample_index]
                current_centers = self._residue_centers(
                    X[atom_start:atom_end], M[atom_start:atom_end, residue_start:residue_end]
                )
                current_vectors = residue_p[:, residue_start:residue_end, :].transpose(0, 1)
                updated_residue_states.append(
                    self._gaussian_forward(
                        residue_q[residue_start:residue_end],
                        current_vectors.contiguous(),
                        current_centers,
                        current_surface,
                    )
                )
                gaussian_features.append(self.gaussian.last_fused)
                atom_start = atom_end
                residue_start = residue_end
            residue_q = pt.cat(updated_residue_states, dim=0)
            gaussian_update = residue_q - original_residue_q
            gaussian_feature = pt.cat(gaussian_features, dim=0)
        decoded = pt.cat([residue_q, pt.norm(residue_p, dim=1)], dim=1)
        logits = self._decode_with_semantic_residual(
            decoded, gaussian_update, gaussian_feature
        )
        if self.direct_protein_residual:
            protein_residual = self._direct_protein_residual(
                decoded, gaussian_update, gaussian_feature
            )
            logits = pt.cat(
                [logits[:, :1] + self.protein_residual_scale * protein_residual.unsqueeze(1),
                 logits[:, 1:]],
                dim=1,
            )
        if self.direct_protein_head:
            protein_output = self._direct_protein_output(
                decoded, gaussian_update, gaussian_feature
            )
            logits = pt.cat(
                [protein_output.unsqueeze(1), logits[:, 1:]], dim=1
            )
        return logits
