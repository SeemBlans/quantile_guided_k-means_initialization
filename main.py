import hashlib
import json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.clustering import random_init, kmeans_plus_plus_init, lloyd_kmeans
from src.metrics import PeakRSSSampler, calculate_silhouette
from src.quantile_guided import quantile_guided_init
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Final_Grades_Dataset_Cleaned.xlsx"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
FEATURES = ["PRELIM", "MID-TERM", "SEMI-FINAL", "FINAL"]
K, TRIALS, MAX_ITER, TOL, BASE_SEED, SIL_SAMPLE = 3, 10, 300, 1e-4, 20261023, 5000
SILHOUETTE_SEED = 20261023

def canonicalize_labels(labels):
    """Rename cluster IDs by order of first appearance."""
    mapping = {}
    canonical = []

    for label in labels:
        label = int(label)

        if label not in mapping:
            mapping[label] = len(mapping)

        canonical.append(mapping[label])

    return np.asarray(canonical, dtype=np.int64)


def canonical_partition_hash(labels):
    """Hash the partition independently of cluster ID numbering."""
    canonical = canonicalize_labels(labels)

    return hashlib.sha256(
        canonical.tobytes()
    ).hexdigest()

def main():
    RESULTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Copy Final_Grades_Dataset_Cleaned.xlsx into {DATA_PATH.parent}")
    df = pd.read_excel(DATA_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError(f"Missing feature columns: {missing}; found {list(df.columns)}")
    frame = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Missing, non-numeric, or non-finite feature values detected.")
    X = StandardScaler().fit_transform(frame.to_numpy(dtype=float))
    
    # Select the silhouette evaluation sample once for the entire experiment.
    sil_rng = np.random.default_rng(SILHOUETTE_SEED)
    silhouette_indices = np.sort(
        sil_rng.choice(
            len(X),
            size=min(SIL_SAMPLE, len(X)),
            replace=False
        )
    )

    X_silhouette = X[silhouette_indices]

    methods = {
        "Random": lambda rng, seed: random_init(X, K, rng),
        "K-Means++": lambda rng, seed: kmeans_plus_plus_init(X, K, rng),
        "Quantile-Guided": lambda rng, seed: quantile_guided_init(X, K, seed),
    }
    records, anomalies = [], []
    metadata = {
        "timing_definition": (
            "initialization_seconds measures the initializer call; "
            "clustering_seconds measures Lloyd clustering including final "
            "assignment/SSE pass; silhouette_seconds measures silhouette "
            "calculation; total_seconds includes sampler setup/teardown, "
            "initialization, clustering, and silhouette calculation"
        ),
        "memory_definition": (
            "sampled absolute process RSS peak and approximate peak RSS "
            "increase above per-run baseline; sampling covers initialization, "
            "clustering, and silhouette calculation; short-lived peaks may "
            "be missed and sampler overhead is included"
        ),
        "silhouette": (
            "Euclidean silhouette calculated on standardized features "
            "using one fixed sample selected uniformly without replacement; "
            "the same observations are reused across all methods and trials"
        ),
        "silhouette_sample_size_requested": SIL_SAMPLE,
        "silhouette_sample_size_actual": len(silhouette_indices),
        "silhouette_sampling_seed": SILHOUETTE_SEED,
        "silhouette_sampling_method": (
            "Uniform sampling without replacement; indices sorted; "
            "sample selected once and reused across all methods and trials"
        ),
        "silhouette_sample_indices": silhouette_indices.tolist(),
    }
    
    for trial in range(TRIALS):
        seed = BASE_SEED + trial
        for name, init_fn in methods.items():            
            start = time.perf_counter()
            try:
                rng = np.random.default_rng(seed)
                
                with PeakRSSSampler() as mem:
                    # Initialization
                    init_start = time.perf_counter()
                    centers, init_info = init_fn(rng, seed)
                    init_s = time.perf_counter() - init_start

                    # Lloyd clustering, including final assignment/SSE pass
                    cluster_start = time.perf_counter()
                    fit = lloyd_kmeans(X, centers, MAX_ITER, TOL)
                    cluster_s = time.perf_counter() - cluster_start

                    # Experiment bookkeeping, excluded from clustering time
                    labels = np.asarray(fit["labels"], dtype=np.int64)

                    label_hash = hashlib.sha256(
                        labels.tobytes()
                    ).hexdigest()

                    partition_hash = canonical_partition_hash(labels)

                    # Silhouette: fixed sample and seed across all runs
                    silhouette_start = time.perf_counter()
                    labels_sample = np.asarray(fit["labels"])[silhouette_indices]

                    sil = calculate_silhouette(
                        X_silhouette,
                        labels_sample
                    )
                    silhouette_s = time.perf_counter() - silhouette_start

                    algorithm_s = init_s + cluster_s

                # Includes sampler setup/teardown, initialization,
                # clustering, and one silhouette calculation.
                total_s = time.perf_counter() - start

                init_d = int(
                    init_info.get("initialization_distance_evaluations", 0)
                )
                cluster_d = int(
                    fit["clustering_distance_evaluations"]
                )
                events = fit["empty_cluster_events"]

                records.append({
                    "labels_sha256": label_hash,        
                    "partition_sha256": partition_hash,            
                    "method": name,
                    "trial": trial + 1,
                    "seed": seed,
                    "n_observations": len(X),
                    "n_features": X.shape[1],
                    "K": K,

                    "initialization_seconds": init_s,
                    "clustering_seconds": cluster_s,
                    "algorithm_seconds": algorithm_s,
                    "silhouette_seconds": silhouette_s,
                    "total_seconds": total_s,

                    "initialization_distance_evaluations": init_d,
                    "clustering_distance_evaluations": cluster_d,
                    "total_distance_evaluations": init_d + cluster_d,

                    "iterations": fit["iterations"],
                    "converged": fit["converged"],

                    # Absolute process memory and approximate increase
                    # above the baseline captured at sampler startup.
                    "baseline_rss_bytes": mem.baseline_bytes,
                    "peak_rss_bytes_sampled": mem.peak_bytes,
                    "peak_rss_mib_sampled": mem.peak_bytes / 1048576,
                    "peak_rss_increase_bytes_sampled":
                        mem.peak_increase_bytes,
                    "peak_rss_increase_mib_sampled":
                        mem.peak_increase_bytes / 1048576,

                    "sse_inertia": fit["inertia"],
                    "silhouette_score": sil,
                    "initial_centers": json.dumps(
                        np.asarray(centers).tolist()
                    ),
                    "initialization_diagnostics": json.dumps(
                        init_info, default=str
                    ),
                    "empty_cluster_events": json.dumps(events),
                    "anomaly_count": len(events),
                    "status": "success",
                    "error": ""
                })

                for event in events:
                    anomalies.append({
                        "method": name,
                        "trial": trial + 1,
                        "seed": seed,
                        "anomaly_type": "empty_cluster",
                        "details": json.dumps(event),
                        "handling": (
                            "retained previous centroid; investigate effect"
                        )
                    })

                print(
                    f"[OK] {name} trial {trial + 1}/{TRIALS}: "
                    f"iter={fit['iterations']}, "
                    f"SSE={fit['inertia']:.4f}, "
                    f"distances={init_d + cluster_d}"
                )
            except NotImplementedError as exc:
                records.append({"method": name, "trial": trial+1, "seed": seed,
                    "K": K, "status": "pending_method_definition", "error": str(exc),
                    "total_seconds": time.perf_counter()-start})
                anomalies.append({"method": name, "trial": trial+1, "seed": seed,
                    "anomaly_type": "method_not_finalized", "details": str(exc),
                    "handling": "do not compare until algorithm is approved"})
                print(f"[PENDING] {name}: finalize its multivariate algorithm first.")
                break
            except Exception as exc:
                records.append({"method": name, "trial": trial+1, "seed": seed,
                    "K": K, "status": "error", "error": repr(exc)})
                anomalies.append({"method": name, "trial": trial+1, "seed": seed,
                    "anomaly_type": "runtime_error", "details": repr(exc),
                    "traceback": traceback.format_exc(limit=5)})
                print(f"[ERROR] {name} trial {trial+1}: {exc}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(records).to_csv(RESULTS / f"experiment_results_{stamp}.csv", index=False)
    pd.DataFrame(anomalies).to_csv(LOGS / f"anomalies_{stamp}.csv", index=False)
    (LOGS / f"run_metadata_{stamp}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    successful = pd.DataFrame([r for r in records if r.get("status") == "success"])
    if not successful.empty:
        
        summary = successful.groupby("method").agg(
            successful_runs=("status", "count"),

            mean_initialization_seconds=("initialization_seconds", "mean"),
            std_initialization_seconds=("initialization_seconds", "std"),

            mean_clustering_seconds=("clustering_seconds", "mean"),
            std_clustering_seconds=("clustering_seconds", "std"),

            mean_algorithm_seconds=("algorithm_seconds", "mean"),
            std_algorithm_seconds=("algorithm_seconds", "std"),

            mean_silhouette_seconds=("silhouette_seconds", "mean"),

            mean_total_seconds=("total_seconds", "mean"),
            std_total_seconds=("total_seconds", "std"),

            mean_total_distance_evaluations=(
                "total_distance_evaluations", "mean"
            ),
            mean_iterations=("iterations", "mean"),

            mean_peak_rss_mib=("peak_rss_mib_sampled", "mean"),
            mean_peak_rss_increase_mib=(
                "peak_rss_increase_mib_sampled", "mean"
            ),

            mean_sse_inertia=("sse_inertia", "mean"),
            std_sse_inertia=("sse_inertia", "std"),
            mean_silhouette_score=("silhouette_score", "mean"),
            std_silhouette_score=("silhouette_score", "std")
        ).reset_index()

        summary.to_csv(RESULTS / f"experiment_summary_{stamp}.csv", index=False)
    print(
        "\nExperiment run finished. Check the result CSV, summary CSV, "
        "run metadata, and anomaly log before interpreting the results."
    )

if __name__ == "__main__":
    main()
