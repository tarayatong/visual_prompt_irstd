"""Support-only spherical clustering and per-direction projection removal."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def spherical_kmeans(features, clusters=64, iterations=40, seed=111, chunk_size=8192):
    x = F.normalize(features.float(), dim=1)
    if not torch.isfinite(x).all() or x.shape[0] < clusters:
        raise ValueError("Clustering needs finite features and at least p background cells")
    generator = torch.Generator(device=x.device).manual_seed(seed)
    # Farthest-cosine initialization makes the first partition reproducible.
    first = int(torch.randint(x.shape[0], (1,), generator=generator, device=x.device).item())
    centers = [x[first]]
    closest_similarity = x @ centers[0]
    for _ in range(1, clusters):
        centers.append(x[int(torch.argmin(closest_similarity).item())])
        closest_similarity = torch.maximum(closest_similarity, x @ centers[-1])
    centers = torch.stack(centers)
    history = []
    for iteration in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(clusters, device=x.device, dtype=torch.long)
        objective = 0.0
        worst_score, worst_index = float('inf'), 0
        for start in range(0, x.shape[0], chunk_size):
            block = x[start:start+chunk_size]
            scores, labels = (block @ centers.T).max(dim=1)
            sums.index_add_(0, labels, block)
            counts += torch.bincount(labels, minlength=clusters)
            objective += float(scores.sum().item())
            if float(scores.min().item()) < worst_score:
                worst_score = float(scores.min().item())
                worst_index = start + int(scores.argmin().item())
        updated = F.normalize(sums, dim=1)
        empty = counts == 0
        if empty.any():
            updated[empty] = x[worst_index]
        shift = float((1 - (centers * updated).sum(dim=1)).clamp_min(0).max().item())
        history.append({'iteration':iteration+1, 'mean_assigned_cosine':objective/x.shape[0], 'max_cosine_shift':shift})
        centers = updated
        if shift < 1e-5:
            break
    return centers, {'background_cells':x.shape[0], 'feature_channels':x.shape[1],
                     'clusters':clusters, 'iterations':len(history), 'cluster_counts':counts.cpu().tolist(), 'history':history}


@torch.no_grad()
def subtract_direction_projections(target, prototypes, mode="sum"):
    t = F.normalize(target.float().reshape(1, -1), dim=1)
    u = F.normalize(prototypes.float(), dim=1)
    if mode == "sum":
        projected = (t @ u.T) @ u
    elif mode == "orthogonal":
        _, singular, vh = torch.linalg.svd(u, full_matrices=False)
        basis = vh[singular > singular.max()*1e-5]
        projected = (t @ basis.T) @ basis
    else:
        raise ValueError(f"Unknown projection mode {mode}")
    residual = t - projected
    norm = float(residual.norm().item())
    if norm < 1e-8:
        raise ValueError("Background directions remove the entire target; cannot normalize the residual")
    corrected = F.normalize(residual, dim=1)
    gram = u @ u.T
    off_diagonal = gram[~torch.eye(len(u), device=u.device, dtype=torch.bool)]
    return corrected, {'projection_mode':mode, 'projection_norm':float(projected.norm().item()),
                       'residual_norm':norm, 'target_score_cos':float((t*corrected).sum().item()),
                       'mean_prototype_pair_cosine':float(off_diagonal.mean().item()),
                       'max_corrected_background_cosine':float((corrected@u.T).max().item())}
