import numpy as np

def random_init(X, k, rng):
    if len(X) < k:
        raise ValueError("Number of observations must be at least K.")
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy(), {"selected_row_indices_zero_based": idx.tolist(),
                           "initialization_distance_evaluations": 0}

def kmeans_plus_plus_init(X, k, rng):
    n = len(X)
    if n < k:
        raise ValueError("Number of observations must be at least K.")
    chosen = [int(rng.integers(n))]
    centers = [X[chosen[0]].copy()]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    count = n
    while len(centers) < k:
        total = float(closest_sq.sum())
        if not np.isfinite(total) or total <= 0:
            remaining = np.setdiff1d(np.arange(n), np.array(chosen))
            if len(remaining) == 0:
                raise ValueError("No unselected observation remains.")
            nxt = int(rng.choice(remaining))
        else:
            nxt = int(rng.choice(n, p=closest_sq / total))
            if nxt in chosen:
                remaining = np.setdiff1d(np.arange(n), np.array(chosen))
                if len(remaining) == 0:
                    raise ValueError("No unselected observation remains.")
                nxt = int(rng.choice(remaining))
        chosen.append(nxt)
        centers.append(X[nxt].copy())
        # Update nearest-center distances only when another center still
        # needs to be sampled. Computing distances to the final selected
        # center is unnecessary for initialization.
        if len(centers) < k:
            new_sq = np.sum((X - X[nxt]) ** 2, axis=1)
            count += n
            closest_sq = np.minimum(closest_sq, new_sq)
    return np.asarray(centers), {
        "selected_row_indices_zero_based": chosen,
        "initialization_distance_evaluations": int(count)
    }

def squared_distances(X, centers):
    delta = X[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", delta, delta)

def lloyd_kmeans(X, initial_centers, max_iter=300, tol=1e-4):
    centers = np.asarray(initial_centers, dtype=float).copy()
    n, k = len(X), len(centers)
    distance_count, events = 0, []
    converged, iterations = False, 0
    for it in range(1, max_iter + 1):
        d2 = squared_distances(X, centers)
        distance_count += n * k
        labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members):
                new_centers[j] = members.mean(axis=0)
            else:
                events.append({"iteration": it, "cluster": j,
                               "event": "empty_cluster",
                               "action": "retain_previous_centroid"})
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        iterations = it
        if shift <= tol:
            converged = True
            break
    final_d2 = squared_distances(X, centers)
    distance_count += n * k
    labels = np.argmin(final_d2, axis=1)
    inertia = float(np.sum(final_d2[np.arange(n), labels]))
    return {"centers": centers, "labels": labels, "iterations": iterations,
            "converged": converged, "clustering_distance_evaluations": distance_count,
            "inertia": inertia, "empty_cluster_events": events}
