"""Pair- and triplet-conditioned Gaussian surface corrections."""

import math

import torch as pt

from .gaussian_projection import inverse_softplus


def build_pair_gaussian_modules(
    *,
    dim,
    vector_dim,
    graph_views,
    dropout,
    pair_mode,
    pair_topk,
    pair_beta,
    pair_sigma_tangent,
    pair_sigma_normal,
    pair_sigma_mode,
    pair_sigma_min,
    pair_joint_mode,
    pair_fusion_mode,
    pair_content_mode,
    pair_content_weight,
    pair_rho_condition_mode,
    pair_vector_condition_mode,
    pair_single_compete_mode,
    pair_single_rescue_beta,
    pair_beta_mode,
    pair_beta_min,
    pair_beta_max,
    pair_beta_init,
    pair_multiview_mode,
    pair_multiview_beta,
    surface_co_update_mode,
    surface_co_update_layers,
    core_cycles,
    triplet_mode,
    triplet_topk,
    triplet_beta,
    triplet_sigma_tangent,
    triplet_sigma_normal,
    triplet_score_temperature,
    triplet_score_clamp,
    triplet_preselect_topm,
    triplet_beta_mode,
    triplet_beta_min,
    triplet_beta_max,
    triplet_beta_init,
    global_feedback_rounds,
    surface_global_feedback_rounds,
):
    """Construct pair/triplet corrections in checkpoint registration order."""

    modules = {
        "pair": None,
        "pair_deep_beta": None,
        "pair_multiview": None,
        "pair_multiview_mix": None,
        "pair_multiview_gate": None,
        "pair_multiview_norm": None,
        "triplet": None,
        "triplet_deep_beta": None,
    }
    if pair_mode != "none" and float(pair_beta) != 0.0:
        modules["pair"] = PairGaussianResidueToSurfaceCorrection(
            dim=dim,
            geom_dim=6,
            pair_topk=pair_topk,
            sigma_t=pair_sigma_tangent,
            sigma_n=pair_sigma_normal,
            sigma_mode=pair_sigma_mode,
            sigma_min=pair_sigma_min,
            beta=pair_beta,
            mode=pair_mode,
            joint_mode=pair_joint_mode,
            fusion_mode="coattn" if pair_fusion_mode == "coattn" else "residual",
            content_mode=pair_content_mode,
            content_weight=pair_content_weight,
            rho_condition_mode=pair_rho_condition_mode,
            vector_condition_mode=pair_vector_condition_mode,
            single_compete_mode=pair_single_compete_mode,
            single_rescue_beta=pair_single_rescue_beta,
            vector_dim=vector_dim,
            dropout=dropout,
        )
        if pair_beta_mode == "learned_deep":
            beta_init = min(max(pair_beta_init, pair_beta_min + 1e-6), pair_beta_max - 1e-6)
            beta_ratio = (beta_init - pair_beta_min) / (pair_beta_max - pair_beta_min)
            beta_logit = math.log(beta_ratio / max(1.0 - beta_ratio, 1e-12))
            deep_rounds = max(
                1,
                surface_co_update_layers - 1 if surface_co_update_mode != "none" else core_cycles - 1,
            )
            modules["pair_deep_beta"] = pt.nn.Parameter(
                pt.full((deep_rounds,), float(beta_logit))
            )
        if pair_multiview_mode != "none" and pair_multiview_beta != 0.0:
            view_count = len(tuple(graph_views))
            modules["pair_multiview"] = pt.nn.ModuleList(
                [
                    PairGaussianResidueToSurfaceCorrection(
                        dim=dim,
                        geom_dim=6,
                        pair_topk=pair_topk,
                        sigma_t=pair_sigma_tangent,
                        sigma_n=pair_sigma_normal,
                        sigma_mode=pair_sigma_mode,
                        sigma_min=pair_sigma_min,
                        beta=pair_beta,
                        mode=pair_mode,
                        joint_mode=pair_joint_mode,
                        fusion_mode="residual",
                        content_mode=pair_content_mode,
                        content_weight=pair_content_weight,
                        rho_condition_mode=pair_rho_condition_mode,
                        vector_condition_mode=pair_vector_condition_mode,
                        single_compete_mode=pair_single_compete_mode,
                        single_rescue_beta=pair_single_rescue_beta,
                        vector_dim=vector_dim,
                        dropout=dropout,
                    )
                    for _ in range(view_count)
                ]
            )
            modules["pair_multiview_mix"] = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * view_count),
                pt.nn.Linear(dim * view_count, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            modules["pair_multiview_gate"] = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + 6),
                pt.nn.Linear(dim * 2 + 6, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
                pt.nn.Sigmoid(),
            )
            modules["pair_multiview_norm"] = pt.nn.LayerNorm(dim)

    if triplet_mode != "none" and float(triplet_beta) != 0.0:
        modules["triplet"] = TripletGaussianResidueToSurfaceCorrection(
            dim=dim,
            geom_dim=6,
            triplet_topk=triplet_topk,
            sigma_t=triplet_sigma_tangent,
            sigma_n=triplet_sigma_normal,
            beta=triplet_beta,
            mode=triplet_mode,
            dropout=dropout,
            score_temperature=triplet_score_temperature,
            score_clamp=triplet_score_clamp,
            preselect_topm=triplet_preselect_topm,
        )
        if triplet_beta_mode == "learned_deep":
            beta_init = min(
                max(triplet_beta_init, triplet_beta_min + 1e-6),
                triplet_beta_max - 1e-6,
            )
            beta_ratio = (beta_init - triplet_beta_min) / (triplet_beta_max - triplet_beta_min)
            beta_logit = math.log(beta_ratio / max(1.0 - beta_ratio, 1e-6))
            deep_rounds = max(
                1,
                core_cycles * (global_feedback_rounds + surface_global_feedback_rounds),
            )
            modules["triplet_deep_beta"] = pt.nn.Parameter(
                pt.full((deep_rounds,), float(beta_logit))
            )
    return modules


class PairGaussianResidueToSurfaceCorrection(pt.nn.Module):
    """Cooperative residue-pair Gaussian correction to a single-residue surface field."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        pair_topk=8,
        sigma_t=8.0,
        sigma_n=4.0,
        sigma_mode="fixed",
        sigma_min=0.5,
        beta=0.05,
        mode="k_nearest",
        joint_mode="independent",
        fusion_mode="residual",
        content_mode="none",
        content_weight=1.0,
        rho_condition_mode="none",
        vector_condition_mode="none",
        single_compete_mode="none",
        single_rescue_beta=0.25,
        vector_dim=16,
        dropout=0.1,
    ):
        super().__init__()
        self.pair_topk = max(2, int(pair_topk))
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.sigma_mode = str(sigma_mode)
        if self.sigma_mode not in {"fixed", "learned", "reuse_single"}:
            raise ValueError("pair Gaussian sigma_mode must be one of: fixed, learned, reuse_single")
        self.sigma_min = float(sigma_min)
        self.beta = float(beta)
        self.mode = str(mode)
        if self.mode not in {"k_nearest", "random"}:
            raise ValueError("pair Gaussian mode must be one of: k_nearest, random")
        self.joint_mode = str(joint_mode)
        if self.joint_mode not in {"independent", "joint_residual", "joint_only"}:
            raise ValueError("pair Gaussian joint_mode must be one of: independent, joint_residual, joint_only")
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"residual", "coattn"}:
            raise ValueError("pair Gaussian fusion_mode must be one of: residual, coattn")
        self.content_mode = str(content_mode)
        if self.content_mode not in {"none", "additive", "gaussian_distance"}:
            raise ValueError("pair Gaussian content_mode must be one of: none, additive, gaussian_distance")
        self.content_weight = float(content_weight)
        self.rho_condition_mode = str(rho_condition_mode)
        if self.rho_condition_mode not in {"none", "surface_dot"}:
            raise ValueError("pair Gaussian rho_condition_mode must be one of: none, surface_dot")
        self.vector_condition_mode = str(vector_condition_mode)
        if self.vector_condition_mode not in {"none", "surface_frame"}:
            raise ValueError("pair Gaussian vector_condition_mode must be one of: none, surface_frame")
        self.single_compete_mode = str(single_compete_mode)
        if self.single_compete_mode not in {"none", "conservative_rescue", "strong_gate"}:
            raise ValueError("single_compete_mode must be one of: none, conservative_rescue, strong_gate")
        self.single_rescue_beta = float(single_rescue_beta)
        self.vector_dim = int(vector_dim)
        self.intervention_mode = "none"
        self.intervention_fraction = 0.25
        self.intervention_distance = 12.0
        self.intervention_seed = 20260816
        self.intervention_residue_indices = ()
        self.intervention_edge_pairs = ()
        self._last_vector_condition_diagnostics = {}
        self.pair_geom_dim = 10 if self.sigma_mode in {"learned", "reuse_single"} else 6
        pair_input_dim = dim * 4 + self.pair_geom_dim
        self.vector_condition_proj = None
        if self.vector_condition_mode == "surface_frame":
            raw_vector_dim = 10 * self.vector_dim
            self.vector_condition_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(raw_vector_dim),
                pt.nn.Linear(raw_vector_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            pair_input_dim += dim
        self.pair_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(pair_input_dim),
            pt.nn.Linear(pair_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.sigma_head = None
        if self.sigma_mode == "learned":
            self.sigma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
            last = self.sigma_head[-1]
            pt.nn.init.zeros_(last.weight)
            base_sigma = pt.as_tensor([self.sigma_t, self.sigma_n], dtype=pt.float32)
            with pt.no_grad():
                last.bias.copy_(inverse_softplus(base_sigma - self.sigma_min))
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.rho_proj = None
        self.rho_surface_proj = None
        rho_input_dim = pair_input_dim
        if self.rho_condition_mode == "surface_dot":
            self.rho_surface_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + geom_dim),
                pt.nn.Linear(dim + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            rho_input_dim += dim + 3
        if self.joint_mode != "independent":
            self.rho_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(rho_input_dim),
                pt.nn.Linear(rho_input_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 3),
                pt.nn.Tanh(),
            )
        self.content_query = None
        self.content_key = None
        self.content_distance_log_weight = None
        if self.content_mode in {"additive", "gaussian_distance"}:
            self.content_query = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
            )
            self.content_key = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + geom_dim),
                pt.nn.Linear(dim + geom_dim, dim),
            )
            if self.content_mode == "gaussian_distance":
                self.content_distance_log_weight = pt.nn.Parameter(pt.zeros(dim))
        self.single_rescue_gate = None
        self.single_compete_gate = None
        if self.single_compete_mode == "conservative_rescue":
            self.single_rescue_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + geom_dim),
                pt.nn.Linear(dim * 2 + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
                pt.nn.Sigmoid(),
            )
            last = self.single_rescue_gate[-2]
            pt.nn.init.zeros_(last.weight)
            pt.nn.init.constant_(last.bias, -2.5)
        elif self.single_compete_mode == "strong_gate":
            self.single_compete_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + geom_dim),
                pt.nn.Linear(dim * 2 + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
        self.coattn_pair_proj = None
        self.coattn_channel = None
        if self.fusion_mode == "coattn":
            self.coattn_pair_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + geom_dim),
                pt.nn.Linear(dim + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.coattn_channel = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + geom_dim),
                pt.nn.Linear(dim * 2 + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
        self.norm = pt.nn.LayerNorm(dim)

    def _pair_indices(self, dist, k):
        nearest = pt.topk(dist, k=k, dim=-1, largest=False).indices
        if self.mode == "random":
            n_res = dist.shape[1]
            offsets = pt.arange(k, device=dist.device).view(1, k)
            base = pt.argmin(dist, dim=-1, keepdim=True)
            nearest = (base + 13 * (offsets + 1) + 17 * pt.arange(dist.shape[0], device=dist.device).view(-1, 1)) % max(n_res, 1)
        pair_i, pair_j = pt.triu_indices(k, k, offset=1, device=dist.device)
        return nearest[:, pair_i], nearest[:, pair_j]

    def _intervene_pair_indices(self, idx_i, idx_j):
        if str(getattr(self, "intervention_mode", "none")) != "shuffle_endpoints":
            return idx_i, idx_j
        if idx_j.shape[-1] < 2:
            return idx_i, idx_j
        shuffled = idx_j.clone()
        unresolved = pt.ones_like(idx_i, dtype=pt.bool)
        for shift in range(1, int(idx_j.shape[-1])):
            candidate = pt.roll(idx_j, shifts=shift, dims=-1)
            usable = unresolved & (candidate != idx_i)
            shuffled = pt.where(usable, candidate, shuffled)
            unresolved = unresolved & ~usable
            if not bool(unresolved.any()):
                break
        shuffled = pt.where(unresolved, idx_j, shuffled)
        return idx_i, shuffled

    def _intervention_keep_mask(self, route_score, pair_dist, idx_i, idx_j):
        mode = str(getattr(self, "intervention_mode", "none"))
        if mode not in {
            "mask_top_routes", "mask_random_routes", "mask_long_range", "mask_short_range",
            "mask_longest_routes", "mask_shortest_routes", "mask_residue_set_routes",
            "mask_edge_set_routes",
        }:
            return pt.ones_like(route_score, dtype=pt.bool)
        pair_count = int(route_score.shape[-1])
        if pair_count <= 1:
            return pt.ones_like(route_score, dtype=pt.bool)
        if mode == "mask_residue_set_routes":
            residue_indices = tuple(
                int(index) for index in getattr(self, "intervention_residue_indices", ())
            )
            if not residue_indices:
                return pt.ones_like(route_score, dtype=pt.bool)
            selected = pt.as_tensor(residue_indices, device=idx_i.device, dtype=idx_i.dtype)
            remove = pt.isin(idx_i, selected) | pt.isin(idx_j, selected)
            keep = ~remove
            all_removed = ~keep.any(dim=-1, keepdim=True)
            fallback = route_score.argmax(dim=-1, keepdim=True)
            keep.scatter_(dim=-1, index=fallback, value=True)
            return pt.where(all_removed, keep, ~remove)
        if mode == "mask_edge_set_routes":
            edge_pairs = tuple(
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in getattr(self, "intervention_edge_pairs", ())
                if int(left) != int(right)
            )
            if not edge_pairs:
                return pt.ones_like(route_score, dtype=pt.bool)
            edge_left = pt.minimum(idx_i, idx_j)
            edge_right = pt.maximum(idx_i, idx_j)
            remove = pt.zeros_like(route_score, dtype=pt.bool)
            for left, right in edge_pairs:
                remove = remove | ((edge_left == left) & (edge_right == right))
            keep = ~remove
            all_removed = ~keep.any(dim=-1, keepdim=True)
            fallback = route_score.argmax(dim=-1, keepdim=True)
            keep.scatter_(dim=-1, index=fallback, value=True)
            return pt.where(all_removed, keep, ~remove)
        if mode in {
            "mask_top_routes", "mask_random_routes", "mask_longest_routes", "mask_shortest_routes"
        }:
            fraction = min(max(float(getattr(self, "intervention_fraction", 0.25)), 0.0), 1.0)
            remove_count = min(max(1, int(math.ceil(pair_count * fraction))), pair_count - 1)
            if mode == "mask_top_routes":
                remove = pt.topk(route_score, k=remove_count, dim=-1).indices
            elif mode == "mask_longest_routes":
                remove = pt.topk(pair_dist, k=remove_count, dim=-1).indices
            elif mode == "mask_shortest_routes":
                remove = pt.topk(-pair_dist, k=remove_count, dim=-1).indices
            else:
                seed = int(getattr(self, "intervention_seed", 20260816))
                pseudo = (
                    idx_i.to(dtype=pt.int64) * 73856093
                    + idx_j.to(dtype=pt.int64) * 19349663
                    + seed
                ).remainder(104729)
                remove = pt.topk(pseudo, k=remove_count, dim=-1).indices
            keep = pt.ones_like(route_score, dtype=pt.bool)
            keep.scatter_(dim=-1, index=remove, value=False)
            return keep
        threshold = float(getattr(self, "intervention_distance", 12.0))
        remove = pair_dist >= threshold if mode == "mask_long_range" else pair_dist < threshold
        keep = ~remove
        all_removed = ~keep.any(dim=-1, keepdim=True)
        fallback = route_score.argmax(dim=-1, keepdim=True)
        keep.scatter_(dim=-1, index=fallback, value=True)
        return pt.where(all_removed, keep, ~remove)

    @staticmethod
    def _masked_softmax(score, keep):
        minimum = pt.finfo(score.dtype).min
        return pt.softmax(score.masked_fill(~keep, minimum), dim=-1)

    def _pair_sigmas(self, h_res, idx_i, idx_j, sigma_res_override=None):
        if self.sigma_mode == "reuse_single" and sigma_res_override is not None:
            sigma_res = sigma_res_override.to(device=h_res.device, dtype=h_res.dtype).clamp_min(self.sigma_min)
            return sigma_res, sigma_res[idx_i].clamp_min(self.sigma_min), sigma_res[idx_j].clamp_min(self.sigma_min)
        if self.sigma_mode != "learned" or self.sigma_head is None:
            return None, None, None
        sigma_res = self.sigma_min + pt.nn.functional.softplus(self.sigma_head(h_res))
        return sigma_res, sigma_res[idx_i].clamp_min(self.sigma_min), sigma_res[idx_j].clamp_min(self.sigma_min)

    def _pair_vector_condition(self, residue_vectors, idx_i, idx_j, rel_i, rel_j, surface_normal, pair_vec, pair_dist):
        self._last_vector_condition_diagnostics = {}
        if self.vector_condition_proj is None or residue_vectors is None:
            return None
        if residue_vectors.ndim != 3 or residue_vectors.shape[1] != self.vector_dim or residue_vectors.shape[2] != 3:
            return None
        if idx_i.numel() and int(pt.maximum(idx_i.max(), idx_j.max()).detach().cpu()) >= residue_vectors.shape[0]:
            return None
        res_v = residue_vectors.to(device=rel_i.device, dtype=rel_i.dtype)
        vi = res_v[idx_i]
        vj = res_v[idx_j]
        normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = normal.unsqueeze(1)
        dir_i = rel_i / rel_i.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_j = rel_j / rel_j.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pair_axis = pair_vec / pair_dist.unsqueeze(-1).clamp_min(1e-6)

        def features(v, direction, axis):
            v_norm = v.norm(dim=-1)
            dot_n = (v * normal.unsqueeze(-2)).sum(dim=-1)
            dot_r = (v * direction.unsqueeze(-2)).sum(dim=-1)
            tangent_sq = (v_norm.square() - dot_n.square()).clamp_min(0.0)
            tangent = tangent_sq.sqrt()
            dot_axis = (v * axis.unsqueeze(-2)).sum(dim=-1)
            return pt.cat([v_norm, dot_n, dot_r, tangent, dot_axis], dim=-1), {
                "v_norm": v_norm,
                "dot_n": dot_n,
                "dot_r": dot_r,
                "tangent": tangent,
                "tangent_sq": tangent_sq,
                "dot_axis": dot_axis,
            }

        feat_i, diag_i = features(vi, dir_i, pair_axis)
        feat_j, diag_j = features(vj, dir_j, -pair_axis)
        raw = pt.cat([feat_i, feat_j], dim=-1)
        projected = self.vector_condition_proj(raw)

        tangent_sq = pt.cat([diag_i["tangent_sq"], diag_j["tangent_sq"]], dim=-1)
        tangent = pt.cat([diag_i["tangent"], diag_j["tangent"]], dim=-1)
        v_norm = pt.cat([diag_i["v_norm"], diag_j["v_norm"]], dim=-1)
        dot_n = pt.cat([diag_i["dot_n"], diag_j["dot_n"]], dim=-1)
        dot_r = pt.cat([diag_i["dot_r"], diag_j["dot_r"]], dim=-1)
        dot_axis = pt.cat([diag_i["dot_axis"], diag_j["dot_axis"]], dim=-1)
        tangent_sq_f = tangent_sq.detach().float()
        self._last_vector_condition_diagnostics = {
            "pair_gaussian_vector_raw_abs": raw.detach().float().abs().mean(),
            "pair_gaussian_vector_raw_std": raw.detach().float().std(unbiased=False),
            "pair_gaussian_vector_raw_max_abs": raw.detach().float().abs().max(),
            "pair_gaussian_vector_raw_finite_frac": pt.isfinite(raw.detach()).float().mean(),
            "pair_gaussian_vector_proj_abs": projected.detach().float().abs().mean(),
            "pair_gaussian_vector_proj_std": projected.detach().float().std(unbiased=False),
            "pair_gaussian_vector_proj_max_abs": projected.detach().float().abs().max(),
            "pair_gaussian_vector_proj_finite_frac": pt.isfinite(projected.detach()).float().mean(),
            "pair_gaussian_vector_v_norm_mean": v_norm.detach().float().mean(),
            "pair_gaussian_vector_v_norm_min": v_norm.detach().float().min(),
            "pair_gaussian_vector_v_norm_max": v_norm.detach().float().max(),
            "pair_gaussian_vector_dot_n_abs": dot_n.detach().float().abs().mean(),
            "pair_gaussian_vector_dot_n_max_abs": dot_n.detach().float().abs().max(),
            "pair_gaussian_vector_dot_r_abs": dot_r.detach().float().abs().mean(),
            "pair_gaussian_vector_dot_axis_abs": dot_axis.detach().float().abs().mean(),
            "pair_gaussian_vector_tangent_mean": tangent.detach().float().mean(),
            "pair_gaussian_vector_tangent_min": tangent.detach().float().min(),
            "pair_gaussian_vector_tangent_sq_min": tangent_sq_f.min(),
            "pair_gaussian_vector_tangent_sq_mean": tangent_sq_f.mean(),
            "pair_gaussian_vector_tangent_sq_zero_frac": (tangent_sq_f <= 0.0).float().mean(),
            "pair_gaussian_vector_tangent_sq_lt1em12_frac": (tangent_sq_f <= 1e-12).float().mean(),
            "pair_gaussian_vector_tangent_sq_lt1em8_frac": (tangent_sq_f <= 1e-8).float().mean(),
            "pair_gaussian_vector_tangent_grad_risk_max": (0.5 / tangent_sq_f.clamp_min(1e-30).sqrt()).max(),
        }
        return projected

    @staticmethod
    def _surface_frame(surface_normal):
        normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ref_x = normal.new_tensor([1.0, 0.0, 0.0]).view(1, 3).expand_as(normal)
        ref_y = normal.new_tensor([0.0, 1.0, 0.0]).view(1, 3).expand_as(normal)
        ref = pt.where(normal[:, :1].abs() < 0.9, ref_x, ref_y)
        tangent1 = pt.cross(normal, ref, dim=-1)
        tangent1 = tangent1 / tangent1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tangent2 = pt.cross(normal, tangent1, dim=-1)
        tangent2 = tangent2 / tangent2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return tangent1, tangent2, normal

    def _joint_scores(self, rel_i, rel_j, surface_normal, rho_input, independent_score, sigma_i=None, sigma_j=None):
        if self.joint_mode == "independent" or self.rho_proj is None:
            return independent_score, None, None, None
        t1, t2, normal = self._surface_frame(surface_normal)
        basis = [t1.unsqueeze(1), t2.unsqueeze(1), normal.unsqueeze(1)]
        xi = pt.stack([(rel_i * b).sum(dim=-1) for b in basis], dim=-1)
        xj = pt.stack([(rel_j * b).sum(dim=-1) for b in basis], dim=-1)
        if sigma_i is None or sigma_j is None:
            sigma = xi.new_tensor([self.sigma_t, self.sigma_t, self.sigma_n]).clamp_min(1e-6)
            xi = xi / sigma.view(1, 1, 3)
            xj = xj / sigma.view(1, 1, 3)
        else:
            sigma_i3 = pt.stack([sigma_i[..., 0], sigma_i[..., 0], sigma_i[..., 1]], dim=-1).clamp_min(1e-6)
            sigma_j3 = pt.stack([sigma_j[..., 0], sigma_j[..., 0], sigma_j[..., 1]], dim=-1).clamp_min(1e-6)
            xi = xi / sigma_i3
            xj = xj / sigma_j3
        rho = 0.85 * self.rho_proj(rho_input)
        rho2 = rho.square().clamp(max=0.95)
        denom = (1.0 - rho2).clamp_min(1e-4)
        independent_energy = 0.5 * (xi.square() + xj.square()).sum(dim=-1)
        joint_energy = 0.5 * ((xi.square() - 2.0 * rho * xi * xj + xj.square()) / denom).sum(dim=-1)
        log_det_penalty = 0.5 * denom.log().sum(dim=-1)
        independent_score = -independent_energy
        joint_score = -joint_energy - log_det_penalty
        coupling_gain = joint_score - independent_score
        return joint_score, independent_score, coupling_gain, rho

    def forward(
        self,
        h_res,
        residue_centers,
        surface_pos,
        surface_normal,
        surface_geom,
        h_surf_single,
        residue_vectors=None,
        pair_only=False,
        beta_override=None,
        pair_dist_scale=None,
        sigma_res_override=None,
        precomputed_dist=None,
        precomputed_pair_indices=None,
    ):
        if self.beta == 0.0 or h_res.shape[0] < 2:
            self.last_top_record = None
            self.last_top_records = []
            return h_surf_single, {}
        beta_eff = self.beta if beta_override is None else beta_override
        k = min(self.pair_topk, residue_centers.shape[0])
        if precomputed_pair_indices is None:
            dist = precomputed_dist
            if dist is None:
                dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
            idx_i, idx_j = self._pair_indices(dist, k)
        else:
            idx_i, idx_j = precomputed_pair_indices
        idx_i, idx_j = self._intervene_pair_indices(idx_i, idx_j)
        sigma_res, sigma_i, sigma_j = self._pair_sigmas(h_res, idx_i, idx_j, sigma_res_override=sigma_res_override)
        rel_i = residue_centers[idx_i] - surface_pos.unsqueeze(1)
        rel_j = residue_centers[idx_j] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        dn_i = (rel_i * normal).sum(dim=-1).abs()
        dn_j = (rel_j * normal).sum(dim=-1).abs()
        dt_i = (rel_i.square().sum(dim=-1).clamp_min(0.0) - dn_i.square()).clamp_min(0.0).sqrt()
        dt_j = (rel_j.square().sum(dim=-1).clamp_min(0.0) - dn_j.square()).clamp_min(0.0).sqrt()
        pair_vec = residue_centers[idx_i] - residue_centers[idx_j]
        pair_dist = pair_vec.norm(dim=-1).clamp_min(1e-6)
        pair_mid = 0.5 * (residue_centers[idx_i] + residue_centers[idx_j])
        mid_rel = pair_mid - surface_pos.unsqueeze(1)
        mid_dn = (mid_rel * normal).sum(dim=-1).abs()
        mid_dt = (mid_rel.square().sum(dim=-1).clamp_min(0.0) - mid_dn.square()).clamp_min(0.0).sqrt()
        if sigma_i is None or sigma_j is None:
            sigma_i_t = dt_i.new_full(dt_i.shape, max(self.sigma_t, 1e-6))
            sigma_i_n = dn_i.new_full(dn_i.shape, max(self.sigma_n, 1e-6))
            sigma_j_t = dt_j.new_full(dt_j.shape, max(self.sigma_t, 1e-6))
            sigma_j_n = dn_j.new_full(dn_j.shape, max(self.sigma_n, 1e-6))
        else:
            sigma_i_t = sigma_i[..., 0].clamp_min(1e-6)
            sigma_i_n = sigma_i[..., 1].clamp_min(1e-6)
            sigma_j_t = sigma_j[..., 0].clamp_min(1e-6)
            sigma_j_n = sigma_j[..., 1].clamp_min(1e-6)
        sigma_pair_t = (sigma_i_t * sigma_j_t).clamp_min(1e-12).sqrt()
        sigma_pair_n = (sigma_i_n * sigma_j_n).clamp_min(1e-12).sqrt()
        score = -0.5 * (
            (dt_i / sigma_i_t).square()
            + (dn_i / sigma_i_n).square()
            + (dt_j / sigma_j_t).square()
            + (dn_j / sigma_j_n).square()
            + 0.25 * (mid_dt / sigma_pair_t).square()
            + 0.25 * (mid_dn / sigma_pair_n).square()
        )
        weight = pt.softmax(score, dim=-1)
        hi = h_res[idx_i]
        hj = h_res[idx_j]
        if pair_dist_scale is None:
            pair_dist_scale = pair_dist.mean().clamp_min(1e-6)
        else:
            pair_dist_scale = pair_dist_scale.to(device=pair_dist.device, dtype=pair_dist.dtype).clamp_min(1e-6)
        pair_geom_parts = [
            dt_i / sigma_i_t,
            dn_i / sigma_i_n,
            dt_j / sigma_j_t,
            dn_j / sigma_j_n,
            mid_dt / sigma_pair_t,
            pair_dist / pair_dist_scale,
        ]
        if sigma_i is not None and sigma_j is not None:
            base_t = max(self.sigma_t, 1e-6)
            base_n = max(self.sigma_n, 1e-6)
            pair_geom_parts.extend(
                [
                    (sigma_i_t / base_t).clamp_min(1e-6).log(),
                    (sigma_i_n / base_n).clamp_min(1e-6).log(),
                    (sigma_j_t / base_t).clamp_min(1e-6).log(),
                    (sigma_j_n / base_n).clamp_min(1e-6).log(),
                ]
            )
        pair_geom = pt.stack(
            pair_geom_parts,
            dim=-1,
        )
        pair_parts = [hi, hj, hi * hj, (hi - hj).abs(), pair_geom]
        vector_condition = self._pair_vector_condition(
            residue_vectors,
            idx_i,
            idx_j,
            rel_i,
            rel_j,
            surface_normal,
            pair_vec,
            pair_dist,
        )
        if vector_condition is not None:
            pair_parts.append(vector_condition)
        pair_input = pt.cat(pair_parts, dim=-1)
        pair_feat = self.pair_proj(pair_input)
        rho_input = pair_input
        rho_dot_i = None
        rho_dot_j = None
        rho_dot_pair = None
        if self.rho_condition_mode == "surface_dot" and self.rho_surface_proj is not None:
            surface_rho = self.rho_surface_proj(pt.cat([h_surf_single, surface_geom], dim=-1))
            surface_rho_pair = surface_rho.unsqueeze(1).expand_as(hi)
            dot_scale = math.sqrt(max(hi.shape[-1], 1))
            rho_dot_i = (hi * surface_rho_pair).sum(dim=-1, keepdim=True) / dot_scale
            rho_dot_j = (hj * surface_rho_pair).sum(dim=-1, keepdim=True) / dot_scale
            rho_dot_pair = ((hi * hj) * surface_rho_pair).sum(dim=-1, keepdim=True) / dot_scale
            rho_input = pt.cat(
                [
                    pair_input,
                    surface_rho_pair,
                    rho_dot_i,
                    rho_dot_j,
                    rho_dot_pair,
                ],
                dim=-1,
            )
        content_score = None
        if self.content_query is not None and self.content_key is not None and self.content_weight != 0.0:
            pair_q = self.content_query(pair_feat)
            surf_k = self.content_key(pt.cat([h_surf_single, surface_geom], dim=-1))
            if self.content_mode == "gaussian_distance":
                diff = pair_q - surf_k.unsqueeze(1)
                weight_content = pt.nn.functional.softplus(self.content_distance_log_weight).to(device=diff.device, dtype=diff.dtype)
                content_score = -0.5 * (weight_content.view(1, 1, -1) * diff.square()).sum(dim=-1)
                content_score = content_score / math.sqrt(max(pair_q.shape[-1], 1))
            else:
                content_score = (pair_q * surf_k.unsqueeze(1)).sum(dim=-1) / math.sqrt(max(pair_q.shape[-1], 1))
            content_score = self.content_weight * content_score
        joint_score, independent_score, coupling_gain, rho = self._joint_scores(
            rel_i,
            rel_j,
            surface_normal,
            rho_input,
            score,
            sigma_i=sigma_i,
            sigma_j=sigma_j,
        )
        if self.joint_mode == "independent":
            route_score = score if content_score is None else score + content_score
            keep = self._intervention_keep_mask(route_score, pair_dist, idx_i, idx_j)
            weight = self._masked_softmax(route_score, keep)
            pair_ctx = (pair_feat * weight.unsqueeze(-1)).sum(dim=1)
        elif self.joint_mode == "joint_residual":
            route_score = joint_score if content_score is None else joint_score + content_score
            independent_route_score = independent_score if content_score is None else independent_score + content_score
            keep = self._intervention_keep_mask(route_score, pair_dist, idx_i, idx_j)
            weight = self._masked_softmax(route_score, keep)
            independent_weight = self._masked_softmax(independent_route_score, keep)
            signed_weight = weight - independent_weight
            pair_ctx = (pair_feat * signed_weight.unsqueeze(-1)).sum(dim=1)
        else:
            route_score = joint_score if content_score is None else joint_score + content_score
            keep = self._intervention_keep_mask(route_score, pair_dist, idx_i, idx_j)
            weight = self._masked_softmax(route_score, keep)
            pair_ctx = (pair_feat * weight.unsqueeze(-1)).sum(dim=1)
        gate = self.gate(pt.cat([pair_ctx, surface_geom], dim=-1))
        base = h_surf_single.new_zeros(h_surf_single.shape) if pair_only else h_surf_single
        if self.fusion_mode == "coattn":
            pair_msg = self.coattn_pair_proj(pt.cat([pair_ctx, surface_geom], dim=-1))
            channel = pt.softmax(self.coattn_channel(pt.cat([base, pair_msg, surface_geom], dim=-1)), dim=-1)
            out = self.norm(channel[:, :1] * base + channel[:, 1:] * pair_msg)
        else:
            channel = None
            pair_msg = beta_eff * gate * pair_ctx
            if pair_only and self.single_compete_mode == "conservative_rescue" and self.single_rescue_gate is not None:
                single_gate = self.single_rescue_gate(pt.cat([h_surf_single, pair_msg, surface_geom], dim=-1))
                out = self.norm(pair_msg + self.single_rescue_beta * single_gate * (h_surf_single - pair_msg))
            elif pair_only and self.single_compete_mode == "strong_gate" and self.single_compete_gate is not None:
                compete = pt.softmax(self.single_compete_gate(pt.cat([h_surf_single, pair_msg, surface_geom], dim=-1)), dim=-1)
                out = self.norm(compete[:, :1] * h_surf_single + compete[:, 1:] * pair_msg)
            else:
                out = self.norm(base + pair_msg)
        entropy = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_top_record = None
        self.last_top_records = []
        if weight.numel() > 0:
            with pt.no_grad():
                w_det = weight.detach().float()
                pair_count = int(w_det.shape[1]) if w_det.ndim == 2 else 1
                diagnostic_topk = max(1, int(getattr(self, "diagnostic_topk", 1)))
                top_count = min(diagnostic_topk, int(w_det.numel()))
                flat_values, flat_indices = pt.topk(w_det.reshape(-1), k=top_count)
                selected = {
                    "weight": flat_values.cpu().tolist(),
                    "flat_index": flat_indices.cpu().tolist(),
                    "residue_i": idx_i.detach().reshape(-1)[flat_indices].cpu().tolist(),
                    "residue_j": idx_j.detach().reshape(-1)[flat_indices].cpu().tolist(),
                    "route_score": route_score.detach().float().reshape(-1)[flat_indices].cpu().tolist(),
                    "base_score": score.detach().float().reshape(-1)[flat_indices].cpu().tolist(),
                    "dt_i": dt_i.detach().float().reshape(-1)[flat_indices].cpu().tolist(),
                    "dn_i": dn_i.detach().float().reshape(-1)[flat_indices].cpu().tolist(),
                    "dt_j": dt_j.detach().float().reshape(-1)[flat_indices].cpu().tolist(),
                    "dn_j": dn_j.detach().float().reshape(-1)[flat_indices].cpu().tolist(),
                    "pair_dist": pair_dist.detach().float().reshape(-1)[flat_indices].cpu().tolist(),
                }
                if independent_score is not None:
                    selected["independent_score"] = (
                        independent_score.detach().float().reshape(-1)[flat_indices].cpu().tolist()
                    )
                if coupling_gain is not None:
                    selected["coupling_gain"] = (
                        coupling_gain.detach().float().reshape(-1)[flat_indices].cpu().tolist()
                    )
                if rho is not None:
                    selected["rho_abs_mean_for_pair"] = (
                        rho.detach().float().abs().mean(dim=-1).reshape(-1)[flat_indices].cpu().tolist()
                    )
                for position, flat_index in enumerate(selected["flat_index"]):
                    surf_idx = int(flat_index) // max(pair_count, 1)
                    pair_rank = int(flat_index) % max(pair_count, 1)
                    record = {
                        "module": "pair_gaussian",
                        "surface_index": int(surf_idx),
                        "pair_rank": int(pair_rank),
                        "weight": float(selected["weight"][position]),
                        "residue_i": int(selected["residue_i"][position]),
                        "residue_j": int(selected["residue_j"][position]),
                        "route_score": float(selected["route_score"][position]),
                        "base_score": float(selected["base_score"][position]),
                        "dt_i": float(selected["dt_i"][position]),
                        "dn_i": float(selected["dn_i"][position]),
                        "dt_j": float(selected["dt_j"][position]),
                        "dn_j": float(selected["dn_j"][position]),
                        "pair_dist": float(selected["pair_dist"][position]),
                    }
                    if independent_score is not None:
                        record["independent_score"] = float(selected["independent_score"][position])
                    if coupling_gain is not None:
                        record["coupling_gain"] = float(selected["coupling_gain"][position])
                    if rho is not None:
                        record["rho_abs_mean_for_pair"] = float(
                            selected["rho_abs_mean_for_pair"][position]
                        )
                    self.last_top_records.append(record)
                self.last_top_record = self.last_top_records[0]
        diag = {
            "pair_gaussian_entropy": entropy.detach(),
            "pair_gaussian_weight_max": weight.max(dim=-1).values.mean().detach(),
            "pair_gaussian_beta_eff": pt.as_tensor(beta_eff, device=h_res.device, dtype=h_res.dtype).detach(),
            "pair_gaussian_intervention_removed_fraction": (~keep).float().mean().detach(),
        }
        if sigma_res is not None:
            diag.update(
                {
                    "pair_gaussian_sigma_tangent_mean": sigma_res[:, 0].mean().detach(),
                    "pair_gaussian_sigma_tangent_min": sigma_res[:, 0].min().detach(),
                    "pair_gaussian_sigma_tangent_max": sigma_res[:, 0].max().detach(),
                    "pair_gaussian_sigma_tangent_std": sigma_res[:, 0].std(unbiased=False).detach(),
                    "pair_gaussian_sigma_normal_mean": sigma_res[:, 1].mean().detach(),
                    "pair_gaussian_sigma_normal_min": sigma_res[:, 1].min().detach(),
                    "pair_gaussian_sigma_normal_max": sigma_res[:, 1].max().detach(),
                    "pair_gaussian_sigma_normal_std": sigma_res[:, 1].std(unbiased=False).detach(),
                }
            )
        if channel is not None:
            diag["pair_gaussian_coattn_pair_weight"] = channel[:, 1].mean().detach()
            diag["pair_gaussian_coattn_single_weight"] = channel[:, 0].mean().detach()
        if rho is not None:
            diag["pair_gaussian_joint_rho_abs"] = rho.abs().mean().detach()
        if vector_condition is not None:
            diag["pair_gaussian_vector_condition_abs"] = vector_condition.abs().mean().detach()
            diag["pair_gaussian_vector_condition_std"] = vector_condition.std(unbiased=False).detach()
            diag.update(getattr(self, "_last_vector_condition_diagnostics", {}))
        if rho_dot_i is not None:
            diag["pair_gaussian_rho_dot_i_abs"] = rho_dot_i.abs().mean().detach()
            diag["pair_gaussian_rho_dot_j_abs"] = rho_dot_j.abs().mean().detach()
            diag["pair_gaussian_rho_dot_pair_abs"] = rho_dot_pair.abs().mean().detach()
        if coupling_gain is not None:
            diag["pair_gaussian_coupling_abs"] = coupling_gain.abs().mean().detach()
            diag["pair_gaussian_coupling_mean"] = coupling_gain.mean().detach()
        if content_score is not None:
            diag["pair_gaussian_content_score_mean"] = content_score.mean().detach()
            diag["pair_gaussian_content_score_std"] = content_score.std(unbiased=False).detach()
            diag["pair_gaussian_content_score_abs"] = content_score.abs().mean().detach()
            if self.content_mode == "gaussian_distance" and self.content_distance_log_weight is not None:
                diag["pair_gaussian_content_distance_weight_mean"] = pt.nn.functional.softplus(
                    self.content_distance_log_weight
                ).mean().detach()
        if self.single_compete_mode == "conservative_rescue" and pair_only and self.single_rescue_gate is not None:
            diag["pair_gaussian_single_rescue_gate_mean"] = single_gate.mean().detach()
            diag["pair_gaussian_single_rescue_beta"] = h_res.new_tensor(self.single_rescue_beta).detach()
        if self.single_compete_mode == "strong_gate" and pair_only and self.single_compete_gate is not None:
            diag["pair_gaussian_strong_gate_single_mean"] = compete[:, 0].mean().detach()
            diag["pair_gaussian_strong_gate_pair_mean"] = compete[:, 1].mean().detach()
        return out, diag

class TripletGaussianResidueToSurfaceCorrection(pt.nn.Module):
    """Three-residue joint Gaussian correction to the residue-to-surface field."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        triplet_topk=8,
        sigma_t=8.0,
        sigma_n=4.0,
        beta=0.03,
        mode="k_nearest",
        dropout=0.1,
        score_temperature=1.0,
        score_clamp=0.0,
        preselect_topm=0,
    ):
        super().__init__()
        self.triplet_topk = max(3, int(triplet_topk))
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.beta = float(beta)
        self.score_temperature = max(float(score_temperature), 1.0e-6)
        self.score_clamp = max(float(score_clamp), 0.0)
        self.preselect_topm = max(0, int(preselect_topm))
        self.mode = str(mode)
        if self.mode not in {"k_nearest", "random"}:
            raise ValueError("triplet Gaussian mode must be one of: k_nearest, random")
        self.triplet_geom_dim = 12
        triplet_input_dim = dim * 9 + self.triplet_geom_dim
        self.triplet_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(triplet_input_dim),
            pt.nn.Linear(triplet_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.rho_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(triplet_input_dim),
            pt.nn.Linear(triplet_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 9),
        )
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def _triplet_indices(self, dist, k):
        nearest = pt.topk(dist, k=k, dim=-1, largest=False).indices
        if self.mode == "random":
            n_res = dist.shape[1]
            offsets = pt.arange(k, device=dist.device).view(1, k)
            base = pt.argmin(dist, dim=-1, keepdim=True)
            nearest = (base + 11 * (offsets + 1) + 19 * pt.arange(dist.shape[0], device=dist.device).view(-1, 1)) % max(n_res, 1)
        combos = pt.combinations(pt.arange(k, device=dist.device), r=3)
        return nearest[:, combos[:, 0]], nearest[:, combos[:, 1]], nearest[:, combos[:, 2]]

    @staticmethod
    def _stable_triplet_corr(raw):
        raw = 0.85 * pt.tanh(raw).view(*raw.shape[:-1], 3, 3)
        r12 = raw[..., 0]
        r13 = raw[..., 1]
        partial23 = raw[..., 2]
        r23 = partial23 * ((1.0 - r12.square()).clamp_min(1e-4) * (1.0 - r13.square()).clamp_min(1e-4)).sqrt() + r12 * r13
        eye = pt.eye(3, device=raw.device, dtype=raw.dtype).view(*([1] * (raw.ndim - 1)), 3, 3)
        corr = eye.expand(*raw.shape[:-1], 3, 3).clone()
        corr[..., 0, 1] = corr[..., 1, 0] = r12
        corr[..., 0, 2] = corr[..., 2, 0] = r13
        corr[..., 1, 2] = corr[..., 2, 1] = r23
        corr = corr + 1e-4 * eye
        return corr, pt.stack([r12, r13, r23], dim=-1)

    @staticmethod
    def _joint_triplet_score(rel_i, rel_j, rel_k, surface_normal, rho_raw, sigma_t, sigma_n):
        # cuSOLVER Cholesky is not implemented for fp16; keep this tiny
        # covariance solve in fp32 under AMP and cast scores back afterwards.
        out_dtype = rel_i.dtype
        rel_i_f = rel_i.float()
        rel_j_f = rel_j.float()
        rel_k_f = rel_k.float()
        surface_normal_f = surface_normal.float()
        rho_raw_f = rho_raw.float()
        t1, t2, normal = PairGaussianResidueToSurfaceCorrection._surface_frame(surface_normal_f)
        basis = [t1.unsqueeze(1), t2.unsqueeze(1), normal.unsqueeze(1)]
        coords = []
        for rel in (rel_i_f, rel_j_f, rel_k_f):
            coords.append(pt.stack([(rel * b).sum(dim=-1) for b in basis], dim=-1))
        sigma = rel_i_f.new_tensor([max(sigma_t, 1e-6), max(sigma_t, 1e-6), max(sigma_n, 1e-6)])
        xi, xj, xk = [c / sigma.view(1, 1, 3) for c in coords]
        z = pt.stack([xi, xj, xk], dim=-1)
        corr, rho = TripletGaussianResidueToSurfaceCorrection._stable_triplet_corr(rho_raw_f)
        chol = pt.linalg.cholesky(corr)
        solved = pt.linalg.solve_triangular(chol, z.unsqueeze(-1), upper=False).squeeze(-1)
        joint_energy = 0.5 * solved.square().sum(dim=(-1, -2))
        log_det_penalty = chol.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).log().sum(dim=(-1, -2))
        joint_score = -joint_energy - log_det_penalty
        independent_score = -0.5 * z.square().sum(dim=(-1, -2))
        coupling_gain = joint_score - independent_score
        return joint_score.to(out_dtype), independent_score.to(out_dtype), coupling_gain.to(out_dtype), rho.to(out_dtype)

    def forward(self, h_res, residue_centers, surface_pos, surface_normal, surface_geom, h_surf, beta_override=None):
        if self.beta == 0.0 or h_res.shape[0] < 3:
            self.last_top_record = None
            return h_surf, {}
        beta_eff = self.beta if beta_override is None else beta_override
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.triplet_topk, residue_centers.shape[0])
        idx_i, idx_j, idx_k = self._triplet_indices(dist, k)
        rel_i = residue_centers[idx_i] - surface_pos.unsqueeze(1)
        rel_j = residue_centers[idx_j] - surface_pos.unsqueeze(1)
        rel_k = residue_centers[idx_k] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)

        def split_dist(rel):
            dn = (rel * normal).sum(dim=-1).abs()
            dt = (rel.square().sum(dim=-1).clamp_min(0.0) - dn.square()).clamp_min(0.0).sqrt()
            return dt, dn

        dt_i, dn_i = split_dist(rel_i)
        dt_j, dn_j = split_dist(rel_j)
        dt_k, dn_k = split_dist(rel_k)
        centroid = (residue_centers[idx_i] + residue_centers[idx_j] + residue_centers[idx_k]) / 3.0
        centroid_dt, centroid_dn = split_dist(centroid - surface_pos.unsqueeze(1))
        dij = (residue_centers[idx_i] - residue_centers[idx_j]).norm(dim=-1)
        dik = (residue_centers[idx_i] - residue_centers[idx_k]).norm(dim=-1)
        djk = (residue_centers[idx_j] - residue_centers[idx_k]).norm(dim=-1)
        scale = pt.stack([dij, dik, djk], dim=-1).mean().clamp_min(1e-6)
        sigma_t = max(self.sigma_t, 1e-6)
        sigma_n = max(self.sigma_n, 1e-6)
        geom = pt.stack(
            [
                dt_i / sigma_t,
                dn_i / sigma_n,
                dt_j / sigma_t,
                dn_j / sigma_n,
                dt_k / sigma_t,
                dn_k / sigma_n,
                centroid_dt / sigma_t,
                centroid_dn / sigma_n,
                dij / scale,
                dik / scale,
                djk / scale,
                (dij + dik + djk) / (3.0 * scale),
            ],
            dim=-1,
        )
        raw_triplet_count = idx_i.shape[1]
        if self.preselect_topm > 0 and raw_triplet_count > self.preselect_topm:
            v_ij = residue_centers[idx_j] - residue_centers[idx_i]
            v_ik = residue_centers[idx_k] - residue_centers[idx_i]
            triangle_area = 0.5 * pt.linalg.cross(v_ij.float(), v_ik.float(), dim=-1).norm(dim=-1).to(dtype=geom.dtype)
            min_pair_dist = pt.stack([dij, dik, djk], dim=-1).min(dim=-1).values
            centroid_prior = -0.5 * ((centroid_dt / sigma_t) ** 2 + (centroid_dn / sigma_n) ** 2)
            area_term = pt.log1p(triangle_area / scale.square().clamp_min(1e-6))
            spread_term = pt.log1p(min_pair_dist / scale.clamp_min(1e-6))
            pre_score = centroid_prior + 0.10 * area_term + 0.05 * spread_term
            keep = min(self.preselect_topm, raw_triplet_count)
            keep_idx = pt.topk(pre_score, k=keep, dim=-1, largest=True).indices

            def gather_triplet(values):
                if values.ndim == 2:
                    return values.gather(1, keep_idx)
                expand_idx = keep_idx.view(*keep_idx.shape, *([1] * (values.ndim - 2))).expand(-1, -1, *values.shape[2:])
                return values.gather(1, expand_idx)

            idx_i = gather_triplet(idx_i)
            idx_j = gather_triplet(idx_j)
            idx_k = gather_triplet(idx_k)
            rel_i = gather_triplet(rel_i)
            rel_j = gather_triplet(rel_j)
            rel_k = gather_triplet(rel_k)
            dt_i = gather_triplet(dt_i)
            dn_i = gather_triplet(dn_i)
            dt_j = gather_triplet(dt_j)
            dn_j = gather_triplet(dn_j)
            dt_k = gather_triplet(dt_k)
            dn_k = gather_triplet(dn_k)
            centroid_dt = gather_triplet(centroid_dt)
            centroid_dn = gather_triplet(centroid_dn)
            dij = gather_triplet(dij)
            dik = gather_triplet(dik)
            djk = gather_triplet(djk)
            geom = gather_triplet(geom)
        hi = h_res[idx_i]
        hj = h_res[idx_j]
        hk = h_res[idx_k]
        triplet_input = pt.cat(
            [hi, hj, hk, hi * hj, hi * hk, hj * hk, (hi - hj).abs(), (hi - hk).abs(), (hj - hk).abs(), geom],
            dim=-1,
        )
        triplet_feat = self.triplet_proj(triplet_input)
        rho_raw = self.rho_proj(triplet_input)
        score, independent_score, coupling_gain, rho = self._joint_triplet_score(rel_i, rel_j, rel_k, surface_normal, rho_raw, sigma_t, sigma_n)
        score_for_weight = score
        if self.score_clamp > 0.0:
            clamp = float(self.score_clamp)
            score_for_weight = score_for_weight.clamp(min=-clamp, max=clamp)
        weight = pt.softmax(score_for_weight / self.score_temperature, dim=-1)
        triplet_ctx = (triplet_feat * weight.unsqueeze(-1)).sum(dim=1)
        gate = self.gate(pt.cat([triplet_ctx, surface_geom], dim=-1))
        out = self.norm(h_surf + beta_eff * gate * triplet_ctx)
        entropy = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_top_record = None
        if weight.numel() > 0:
            with pt.no_grad():
                w_det = weight.detach().float()
                flat_idx = int(w_det.reshape(-1).argmax().cpu().item())
                triplet_count = int(w_det.shape[1]) if w_det.ndim == 2 else 1
                surf_idx = flat_idx // max(triplet_count, 1)
                triplet_rank = flat_idx % max(triplet_count, 1)
                self.last_top_record = {
                    "module": "triplet_gaussian",
                    "surface_index": int(surf_idx),
                    "triplet_rank": int(triplet_rank),
                    "weight": float(w_det[surf_idx, triplet_rank].cpu().item()),
                    "residue_i": int(idx_i.detach()[surf_idx, triplet_rank].cpu().item()),
                    "residue_j": int(idx_j.detach()[surf_idx, triplet_rank].cpu().item()),
                    "residue_k": int(idx_k.detach()[surf_idx, triplet_rank].cpu().item()),
                    "score_for_weight": float(score_for_weight.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "joint_score": float(score.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "independent_score": float(independent_score.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "coupling_gain": float(coupling_gain.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "rho_abs_mean_for_triplet": float(rho.detach().float()[surf_idx, triplet_rank].abs().mean().cpu().item()),
                    "dt_i": float(dt_i.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dn_i": float(dn_i.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dt_j": float(dt_j.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dn_j": float(dn_j.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dt_k": float(dt_k.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dn_k": float(dn_k.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dij": float(dij.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dik": float(dik.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "djk": float(djk.detach().float()[surf_idx, triplet_rank].cpu().item()),
                }
        diag = {
            "triplet_gaussian_entropy": entropy.detach(),
            "triplet_gaussian_weight_max": weight.max(dim=-1).values.mean().detach(),
            "triplet_gaussian_beta_eff": pt.as_tensor(beta_eff, device=h_res.device, dtype=h_res.dtype).detach(),
            "triplet_gaussian_rho_abs": rho.abs().mean().detach(),
            "triplet_gaussian_coupling_abs": coupling_gain.abs().mean().detach(),
            "triplet_gaussian_coupling_mean": coupling_gain.mean().detach(),
            "triplet_gaussian_independent_score_mean": independent_score.mean().detach(),
            "triplet_gaussian_joint_score_mean": score.mean().detach(),
            "triplet_gaussian_score_for_weight_mean": score_for_weight.mean().detach(),
            "triplet_gaussian_score_temperature": h_res.new_tensor(float(self.score_temperature)).detach(),
            "triplet_gaussian_score_clamp": h_res.new_tensor(float(self.score_clamp)).detach(),
            "triplet_gaussian_gate_mean": gate.mean().detach(),
            "triplet_gaussian_update_norm": triplet_ctx.norm(dim=-1).mean().detach(),
            "triplet_gaussian_triplet_count": h_res.new_tensor(float(idx_i.shape[1])).detach(),
            "triplet_gaussian_raw_triplet_count": h_res.new_tensor(float(raw_triplet_count)).detach(),
            "triplet_gaussian_preselect_topm": h_res.new_tensor(float(self.preselect_topm)).detach(),
        }
        return out, diag
