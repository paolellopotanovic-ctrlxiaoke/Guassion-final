import hashlib
import math
from pathlib import Path

import torch as pt

try:
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    np = None
    csr_matrix = None
    dijkstra = None
    cKDTree = None

from .dataset import cache_key_for_row

atom_chemistry_from_q = None
radius_from_legacy_q = None


def s1_surface_cache_key_for_row(row, num_surface_points=1024, graph_k=16, lappe_dim=0, version="s1_surface_proxy_v1"):
    raw = "|".join(
        [
            cache_key_for_row(row),
            str(version),
            f"p={int(num_surface_points)}",
            f"gk={int(graph_k)}",
            f"lappe={int(lappe_dim)}",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def s1_surface_cache_path_for_row(cache_dir, row, num_surface_points=1024, graph_k=16, lappe_dim=0, version="s1_surface_proxy_v1"):
    return Path(cache_dir) / s1_surface_cache_key_for_row(
        row,
        num_surface_points=num_surface_points,
        graph_k=graph_k,
        lappe_dim=lappe_dim,
        version=version,
    )


def residue_centers_from_item(item):
    X = item["X"].float()
    M = item["M"].float()
    weights = M.to(dtype=X.dtype)
    denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    return weights.transpose(0, 1).matmul(X) / denom


def _farthest_point_sample(points, count):
    n = int(points.shape[0])
    count = min(int(count), n)
    if count <= 0:
        return pt.empty((0,), dtype=pt.long)
    if np is not None and points.device.type == "cpu":
        values = points.detach().numpy().astype(np.float32, copy=False)
        center = values.mean(axis=0, keepdims=True)
        first = int(np.argmax(np.sum((values - center) ** 2, axis=1)))
        selected = np.empty((count,), dtype=np.int64)
        selected[0] = first
        min_dist = np.sum((values - values[first]) ** 2, axis=1)
        for index in range(1, count):
            choice = int(np.argmax(min_dist))
            selected[index] = choice
            delta = values - values[choice]
            min_dist = np.minimum(min_dist, np.sum(delta * delta, axis=1))
        return pt.from_numpy(selected)
    center = points.mean(dim=0, keepdim=True)
    first = int(pt.argmax(((points - center) ** 2).sum(dim=-1)).item())
    selected = [first]
    min_dist = ((points - points[first]) ** 2).sum(dim=-1)
    for _ in range(1, count):
        idx = int(pt.argmax(min_dist).item())
        selected.append(idx)
        min_dist = pt.minimum(min_dist, ((points - points[idx]) ** 2).sum(dim=-1))
    return pt.tensor(selected, dtype=pt.long)


def _normalized_lappe_from_edges(num_nodes, edge_idx, dim):
    dim = int(dim)
    if dim <= 0 or num_nodes <= 1:
        return pt.empty((num_nodes, 0), dtype=pt.float)
    adj = pt.zeros((num_nodes, num_nodes), dtype=pt.float)
    if edge_idx.numel():
        src = pt.arange(num_nodes).unsqueeze(-1).expand_as(edge_idx).reshape(-1)
        dst = edge_idx.reshape(-1)
        valid = (dst >= 0) & (dst < num_nodes)
        adj[src[valid], dst[valid]] = 1.0
        adj[dst[valid], src[valid]] = 1.0
    deg = adj.sum(dim=-1)
    inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    lap = pt.eye(num_nodes) - inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    try:
        eigvals, eigvecs = pt.linalg.eigh(lap)
        order = pt.argsort(eigvals)
        vecs = eigvecs[:, order[1 : dim + 1]]
    except RuntimeError:
        vecs = pt.zeros((num_nodes, min(dim, max(num_nodes - 1, 0))), dtype=pt.float)
    if vecs.shape[1] < dim:
        vecs = pt.cat([vecs, pt.zeros((num_nodes, dim - vecs.shape[1]), dtype=pt.float)], dim=-1)
    return vecs[:, :dim].float()


def _topk_graph_from_distance(distance, graph_k):
    p = int(distance.shape[0])
    graph_k = min(int(graph_k), max(p - 1, 0))
    if p <= 1 or graph_k <= 0:
        return pt.empty((p, 0), dtype=pt.long), pt.empty((p, 0), dtype=pt.float)
    distance = distance.clone()
    distance.fill_diagonal_(float("inf"))
    graph_dist, graph_idx = pt.topk(distance, k=graph_k, dim=-1, largest=False)
    finite = pt.isfinite(graph_dist)
    self_idx = pt.arange(p, device=graph_idx.device).unsqueeze(-1).expand_as(graph_idx)
    graph_idx = pt.where(finite, graph_idx, self_idx)
    graph_dist = pt.where(finite, graph_dist, graph_dist.new_zeros(graph_dist.shape))
    return graph_idx.contiguous(), graph_dist.contiguous()


def _approx_geodesic_distance(surface_pos, surface_normal, local_k=16):
    """Shortest-path distance on a normal-aware local surface point graph."""
    if np is None or cKDTree is None or csr_matrix is None or dijkstra is None:
        raise RuntimeError("Approximate geodesic surface graphs require scipy and numpy.")
    position = surface_pos.detach().cpu().numpy().astype(np.float32, copy=False)
    normal = surface_normal.detach().cpu().numpy().astype(np.float32, copy=False)
    p = int(position.shape[0])
    local_k = min(max(2, int(local_k)), max(p - 1, 1))
    distance, neighbor = cKDTree(position.astype(np.float64, copy=False)).query(
        position, k=local_k + 1, workers=1
    )
    neighbor = np.asarray(neighbor[:, 1:], dtype=np.int64)
    distance = np.asarray(distance[:, 1:], dtype=np.float32)
    source = np.repeat(np.arange(p, dtype=np.int64), neighbor.shape[1])
    target = neighbor.reshape(-1)
    delta = position[target] - position[source]
    length = np.maximum(distance.reshape(-1), np.float32(1e-6))
    direction = delta / length[:, None]
    normal_dot = np.sum(normal[source] * normal[target], axis=1).clip(-1.0, 1.0)
    normal_penalty = 1.0 - normal_dot
    tangent_penalty = np.maximum(
        np.abs(np.sum(direction * normal[source], axis=1)),
        np.abs(np.sum(direction * normal[target], axis=1)),
    )
    weight = length * (1.0 + 2.0 * normal_penalty + 4.0 * tangent_penalty * tangent_penalty)
    adjacency = csr_matrix((weight, (source, target)), shape=(p, p))
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    geodesic = dijkstra(adjacency, directed=False, return_predecessors=False)
    return pt.from_numpy(np.asarray(geodesic, dtype=np.float32))


def _surface_graph(surface_pos, graph_k, surface_normal=None, mode="euclidean"):
    p = int(surface_pos.shape[0])
    graph_k = min(int(graph_k), max(p - 1, 0))
    if p <= 1 or graph_k <= 0:
        return pt.empty((p, 0), dtype=pt.long), pt.empty((p, 0), dtype=pt.float)
    if str(mode) in {"geodesic", "approx_geodesic"}:
        if surface_normal is None:
            raise ValueError("Approximate geodesic surface graphs require surface normals.")
        dist = _approx_geodesic_distance(
            surface_pos, surface_normal, local_k=max(8, min(16, graph_k))
        )
    else:
        dist = pt.cdist(surface_pos, surface_pos).clamp_min(0.0)
    return _topk_graph_from_distance(dist, graph_k)


def _surface_point_target(surface_area, accessible_candidates, point_mode, point_density, maximum):
    maximum = max(1, int(maximum))
    if float(point_density) <= 0.0:
        return maximum
    if str(point_mode) == "area_density":
        target = int(round(float(surface_area) * float(point_density)))
    elif str(point_mode) == "dynamic":
        target = int(round(float(accessible_candidates) * float(point_density)))
    else:
        return maximum
    return max(min(64, maximum), min(maximum, target))


def _atom_radii_from_q(q_atom):
    # Conservative element fallback for cached atom features whose exact layout
    # can differ across APEX cache generations. This is enough for a first
    # Shrake-Rupley-style molecular surface pilot and is clearly marked in cache.
    n = int(q_atom.shape[0])
    if radius_from_legacy_q is not None and q_atom.shape[1] >= 30:
        return radius_from_legacy_q(q_atom.float()).to(dtype=pt.float)
    radii = pt.full((n,), 1.70, dtype=pt.float)
    if q_atom.numel() == 0:
        return radii
    # If the first common element one-hot slots exist, use them; otherwise all-C.
    # Actual PeSTo slot order starts C/O/N/S/P/Se.
    guesses = [
        (0, 1.70),  # C
        (1, 1.52),  # O
        (2, 1.55),  # N
        (3, 1.80),  # S
        (4, 1.80),  # P
        (5, 1.90),  # Se
    ]
    for idx, radius in guesses:
        if q_atom.shape[1] > idx:
            mask = q_atom[:, idx] > 0.5
            radii = pt.where(mask, pt.full_like(radii, radius), radii)
    return radii


def _shrake_accessible_mask_kdtree(candidate_position, atom_position, radii, atom_ids):
    if cKDTree is None or np is None:
        raise RuntimeError("Shrake kdtree mode requires scipy and numpy.")
    candidate_numpy = candidate_position.detach().cpu().numpy().astype(np.float32, copy=False)
    atom_numpy = atom_position.detach().cpu().numpy().astype(np.float32, copy=False)
    radii_numpy = radii.detach().cpu().numpy().astype(np.float32, copy=False)
    atom_ids_numpy = atom_ids.detach().cpu().numpy()
    maximum_radius = float(radii_numpy.max()) + 1e-4
    candidate_tree = cKDTree(candidate_numpy.astype(np.float64, copy=False))
    atom_tree = cKDTree(atom_numpy.astype(np.float64, copy=False))
    pairs = candidate_tree.sparse_distance_matrix(
        atom_tree,
        max_distance=maximum_radius,
        output_type="coo_matrix",
    )
    rows = pairs.row.astype(np.int64, copy=False)
    columns = pairs.col.astype(np.int64, copy=False)
    not_own_atom = columns != atom_ids_numpy[rows]
    rows = rows[not_own_atom]
    columns = columns[not_own_atom]
    delta = candidate_numpy[rows] - atom_numpy[columns]
    distance = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32)).astype(
        np.float32, copy=False
    )
    buried_rows = rows[distance < (radii_numpy[columns] - np.float32(1e-4))]
    accessible = np.ones((candidate_numpy.shape[0],), dtype=np.bool_)
    accessible[buried_rows] = False
    return pt.from_numpy(accessible)


def build_s1_surface_shrake_cache(
    item,
    num_surface_points=1024,
    graph_k=16,
    lappe_dim=0,
    probe_radius=1.4,
    sphere_samples=64,
    neighbor_method="bruteforce",
    point_mode="fixed",
    point_density=0.0,
    graph_mode="euclidean",
    chemistry_mode="none",
    extra_graph_ks=(),
):
    """Build a minimal true-ish molecular surface point cache.

    This uses a Shrake-Rupley-style solvent accessible surface approximation:
    sample points on atom spheres with probe radius, keep points not buried
    inside neighboring atom spheres, then farthest-point sample to the requested
    cache size. It is not MSMS/EDTSurf, but unlike the proxy cache it is built
    from atom-level solvent accessibility rather than residue-center outward
    offsets.
    """
    X = item["X"].float()
    q = item.get("q")
    if q is None:
        q = pt.zeros((X.shape[0], 0), dtype=pt.float)
    else:
        q = q.float()
    centers = residue_centers_from_item(item)
    n_res = int(centers.shape[0])
    n_atom = int(X.shape[0])
    if n_res <= 0 or n_atom <= 0:
        raise ValueError("Cannot build S1 Shrake surface cache for empty structure.")

    sphere_samples = max(int(sphere_samples), 16)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []
    for i in range(sphere_samples):
        z = 1.0 - (2.0 * (i + 0.5) / sphere_samples)
        r = math.sqrt(max(1.0 - z * z, 0.0))
        theta = golden * i
        dirs.append([math.cos(theta) * r, math.sin(theta) * r, z])
    dirs = pt.tensor(dirs, dtype=pt.float)

    radii = _atom_radii_from_q(q) + float(probe_radius)
    cand_pos = X[:, None, :] + dirs[None, :, :] * radii[:, None, None]
    cand_normal = dirs[None, :, :].expand(n_atom, sphere_samples, 3).reshape(-1, 3)
    cand_pos = cand_pos.reshape(-1, 3)

    # Surface point is accessible if it is not inside any other atom's
    # solvent-expanded sphere. Chunking keeps memory predictable.
    atom_ids = pt.arange(n_atom).repeat_interleave(sphere_samples)
    if neighbor_method == "kdtree":
        accessible = _shrake_accessible_mask_kdtree(cand_pos, X, radii, atom_ids)
    elif neighbor_method == "bruteforce":
        accessible = pt.ones((cand_pos.shape[0],), dtype=pt.bool)
        chunk = 4096
        for start in range(0, cand_pos.shape[0], chunk):
            stop = min(start + chunk, cand_pos.shape[0])
            dist = pt.cdist(cand_pos[start:stop], X).clamp_min(0.0)
            inside = dist < (radii[None, :] - 1e-4)
            own = atom_ids[start:stop, None] == pt.arange(n_atom)[None, :]
            inside = inside & ~own
            accessible[start:stop] = ~inside.any(dim=1)
    else:
        raise ValueError(f"Unsupported Shrake neighbor method: {neighbor_method}")

    surface_pos = cand_pos[accessible]
    surface_normal = cand_normal[accessible]
    if surface_pos.shape[0] == 0:
        return build_s1_surface_proxy_cache(
            item,
            num_surface_points=num_surface_points,
            graph_k=graph_k,
            lappe_dim=lappe_dim,
        )

    residue_sasa = pt.zeros((n_res,), dtype=pt.float)
    atom_to_residue = item["M"].float().argmax(dim=1).long()
    atom_area = 4.0 * math.pi * radii.square() / float(sphere_samples)
    residue_sasa.index_add_(0, atom_to_residue, atom_area * accessible.reshape(n_atom, sphere_samples).sum(dim=1).to(dtype=pt.float))

    surface_area = float(residue_sasa.sum().item())
    target = _surface_point_target(
        surface_area, int(surface_pos.shape[0]), point_mode, point_density, num_surface_points
    )
    p = min(target, int(surface_pos.shape[0]))
    selected = _farthest_point_sample(surface_pos, p)
    source_atom_idx = atom_ids[accessible][selected].long().contiguous()
    surface_pos = surface_pos[selected].contiguous()
    surface_normal = surface_normal[selected].contiguous()
    surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dist_to_res = pt.cdist(surface_pos, centers).clamp_min(0.0)
    nearest_dist, nearest_idx = pt.min(dist_to_res, dim=-1)
    centroid = centers.mean(dim=0, keepdim=True)
    radial = (surface_pos - centroid).norm(dim=-1, keepdim=True)
    radial_norm = radial / radial.mean().clamp_min(1e-6)
    density = pt.topk(dist_to_res, k=min(8, n_res), dim=-1, largest=False).values.mean(dim=-1, keepdim=True)
    density = density / density.mean().clamp_min(1e-6)
    nearest_dist_norm = nearest_dist.unsqueeze(-1) / nearest_dist.mean().clamp_min(1e-6)
    geom = pt.cat([surface_normal, radial_norm, density, nearest_dist_norm], dim=-1).float()

    graph_ks = tuple(dict.fromkeys([int(graph_k), *(int(k) for k in extra_graph_ks if int(k) > 0)]))
    maximum_graph_k = max(graph_ks)
    maximum_graph_idx, maximum_graph_dist = _surface_graph(
        surface_pos, maximum_graph_k, surface_normal, mode=graph_mode
    )
    base_width = min(int(graph_k), maximum_graph_idx.shape[1])
    graph_idx = maximum_graph_idx[:, :base_width].contiguous()
    graph_dist = maximum_graph_dist[:, :base_width].contiguous()
    if graph_idx.numel():
        surface_curvature = (1.0 - (surface_normal.unsqueeze(1) * surface_normal[graph_idx]).sum(dim=-1).clamp(-1.0, 1.0)).mean(dim=-1, keepdim=True)
    else:
        surface_curvature = surface_pos.new_zeros((p, 1))
    cache = {
        "surface_pos": surface_pos.float(),
        "surface_normal": surface_normal.float(),
        "surface_geom": geom.float(),
        "surface_graph_idx": graph_idx.long(),
        "surface_graph_dist": graph_dist.float(),
        "surface_lappe": _normalized_lappe_from_edges(p, graph_idx, lappe_dim),
        "nearest_residue_idx": nearest_idx.long(),
        "source_atom_idx": source_atom_idx,
        "surface_curvature": surface_curvature.float(),
        "residue_sasa": residue_sasa.float(),
        "nearest_residue_dist": nearest_dist.float(),
        "surface_mask": pt.ones((p,), dtype=pt.float),
        "num_surface_points": int(p),
        "num_residues": int(n_res),
        "cache_kind": "shrake_rypley_atom_surface_v0",
        "probe_radius": float(probe_radius),
        "sphere_samples": int(sphere_samples),
        "accessible_candidates": int(accessible.sum().item()),
        "surface_graph_mode": str(graph_mode),
        "surface_point_mode": str(point_mode),
        "surface_point_density": float(point_density),
        "surface_point_max": int(num_surface_points),
        "surface_area": surface_area,
        "surface_graph_ks": graph_ks,
        "semantic_schema": "m4_atom_semantics_v1",
    }
    for extra_k in graph_ks:
        width = min(int(extra_k), maximum_graph_idx.shape[1])
        cache[f"surface_graph_idx_k{extra_k}"] = maximum_graph_idx[:, :width].contiguous()
        cache[f"surface_graph_dist_k{extra_k}"] = maximum_graph_dist[:, :width].contiguous()
    if chemistry_mode != "none":
        if atom_chemistry_from_q is not None:
            atom_chem = atom_chemistry_from_q(q, mode=chemistry_mode)
            cache["surface_chemistry"] = atom_chem[source_atom_idx].float()
        cache["surface_element_idx"] = q[source_atom_idx, :30].argmax(dim=-1).long()
    if "y" in item:
        y = item["y"].float()
        cache["proxy_surface_labels"] = y[nearest_idx].float()
    return cache


def build_s1_surface_cache_for_version(
    item,
    version="s1_surface_proxy_v1",
    num_surface_points=1024,
    graph_k=16,
    lappe_dim=0,
    shrake_probe_radius=1.4,
    shrake_sphere_samples=64,
    shrake_neighbor_method="bruteforce",
    point_mode="fixed",
    point_density=0.0,
    graph_mode="euclidean",
    chemistry_mode="none",
    extra_graph_ks=(),
):
    """Dispatch online construction to the builder used by the named disk cache."""
    normalized_version = str(version).strip().lower()
    if "shrake" in normalized_version or "truegeo" in normalized_version or "semantic" in normalized_version:
        return build_s1_surface_shrake_cache(
            item,
            num_surface_points=num_surface_points,
            graph_k=graph_k,
            lappe_dim=lappe_dim,
            probe_radius=shrake_probe_radius,
            sphere_samples=shrake_sphere_samples,
            neighbor_method=shrake_neighbor_method,
            point_mode=point_mode,
            point_density=point_density,
            graph_mode=graph_mode,
            chemistry_mode=chemistry_mode,
            extra_graph_ks=extra_graph_ks,
        )
    if "proxy" in normalized_version:
        return build_s1_surface_proxy_cache(
            item,
            num_surface_points=num_surface_points,
            graph_k=graph_k,
            lappe_dim=lappe_dim,
        )
    raise ValueError(f"Unsupported S1 surface cache version for online construction: {version}")


def build_s1_surface_proxy_cache(item, num_surface_points=1024, graph_k=16, lappe_dim=0, outward_offset=3.0):
    """Build a runnable proxy surface cache from residue centers.

    This is not a true solvent-excluded surface. It is a fallback interface so
    S1 can run before MSMS/EDTSurf/FreeSASA-style true surface caches exist.
    True surface caches should provide the same keys: surface_pos,
    surface_normal, surface_geom, surface_graph_idx, surface_graph_dist.
    """
    centers = residue_centers_from_item(item)
    n_res = int(centers.shape[0])
    if n_res <= 0:
        raise ValueError("Cannot build S1 surface cache for zero residues.")
    p = min(int(num_surface_points), n_res)
    selected = _farthest_point_sample(centers, p)
    anchor = centers[selected].contiguous()
    centroid = centers.mean(dim=0, keepdim=True)
    normal = anchor - centroid
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    surface_pos = anchor + float(outward_offset) * normal

    dist_to_res = pt.cdist(surface_pos, centers).clamp_min(0.0)
    nearest_dist, nearest_idx = pt.min(dist_to_res, dim=-1)
    radial = (surface_pos - centroid).norm(dim=-1, keepdim=True)
    radial_norm = radial / radial.mean().clamp_min(1e-6)
    density = pt.topk(dist_to_res, k=min(8, n_res), dim=-1, largest=False).values.mean(dim=-1, keepdim=True)
    density = density / density.mean().clamp_min(1e-6)
    nearest_dist_norm = nearest_dist.unsqueeze(-1) / nearest_dist.mean().clamp_min(1e-6)
    geom = pt.cat([normal, radial_norm, density, nearest_dist_norm], dim=-1).float()

    graph_idx, graph_dist = _surface_graph(surface_pos, graph_k)
    cache = {
        "surface_pos": surface_pos.float(),
        "surface_normal": normal.float(),
        "surface_geom": geom.float(),
        "surface_graph_idx": graph_idx.long(),
        "surface_graph_dist": graph_dist.float(),
        "surface_lappe": _normalized_lappe_from_edges(p, graph_idx, lappe_dim),
        "nearest_residue_idx": nearest_idx.long(),
        "nearest_residue_dist": nearest_dist.float(),
        "surface_mask": pt.ones((p,), dtype=pt.float),
        "num_surface_points": int(p),
        "num_residues": int(n_res),
        "cache_kind": "proxy_residue_outward_surface",
    }
    if "y" in item:
        y = item["y"].float()
        cache["proxy_surface_labels"] = y[nearest_idx].float()
    return cache


def save_s1_surface_cache(path, cache):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pt.save({k: (v.cpu() if isinstance(v, pt.Tensor) else v) for k, v in cache.items()}, path)
