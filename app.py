from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.clustering import random_init, kmeans_plus_plus_init, lloyd_kmeans
from src.metrics import PeakRSSSampler, calculate_silhouette
from src.quantile_guided import quantile_guided_init

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "Final_Grades_Dataset_Cleaned.xlsx"
FEATURES = ["PRELIM", "MID-TERM", "SEMI-FINAL", "FINAL"]

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
    """Hash a partition independently of cluster ID numbering."""
    canonical = canonicalize_labels(labels)
    return hashlib.sha256(canonical.tobytes()).hexdigest()

st.set_page_config(page_title="Quantile-Guided K-Means Lab", page_icon="📊", layout="wide")

st.title("Quantile-Guided K-Means Research Lab")
st.caption("Experimental dashboard · Random vs K-Means++ vs proposed Quantile-Guided initialization")
st.info(
    "Research status: PCA-score Quantile-Guided initialization is implemented "
    "alongside Random and K-Means++. The proposed method selects complete "
    "observed rows using deterministic PC1-score quantile positions. "
    "Validate the implementation and complete repeated trials before drawing "
    "comparative conclusions."
)

with st.sidebar:
    st.header("Experiment settings")
    k = st.selectbox("Number of clusters (K)", [2, 3, 4, 5, 6], index=1)
    trials = st.select_slider("Repeated trials", options=[1, 3, 5, 10, 20, 30], value=10)
    standardize = st.toggle("Standardize grade features", value=True)
    max_iter = st.number_input("Maximum iterations", min_value=10, max_value=2000, value=300, step=50)
    tol = st.number_input("Centroid-shift tolerance", min_value=0.000001, max_value=0.1,
                          value=0.0001, format="%.6f")
    silhouette_sample = st.select_slider("Silhouette sample size", options=[500, 1000, 2000, 3000], value=1000)
    base_seed = st.number_input("Base random seed", min_value=0, max_value=2147483647, value=20261023)
    st.caption("The same preprocessing, K, stopping settings, and evaluation rules are used for all three methods.")

st.subheader("1. Load dataset")
uploaded = st.file_uploader("Upload cleaned Excel dataset (.xlsx)", type=["xlsx"])
if uploaded is not None:
    df = pd.read_excel(uploaded)
    source_name = uploaded.name
elif DEFAULT_DATA.exists():
    df = pd.read_excel(DEFAULT_DATA)
    source_name = DEFAULT_DATA.name
else:
    df = None
    source_name = ""

if df is None:
    st.warning("Place Final_Grades_Dataset_Cleaned.xlsx in the project's data/ folder, or upload it above.")
    st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}. Found: {list(df.columns)}")
    st.stop()

feature_frame = df[FEATURES].apply(pd.to_numeric, errors="coerce")
bad_rows = feature_frame.index[feature_frame.isna().any(axis=1)].tolist()
if bad_rows:
    st.error(f"{len(bad_rows)} row(s) contain missing or non-numeric feature values. Resolve and document these before running.")
    st.stop()
X_raw = feature_frame.to_numpy(dtype=float)
if not np.isfinite(X_raw).all():
    st.error("Non-finite feature values were found. Resolve and document these before running.")
    st.stop()
if len(X_raw) <= k:
    st.error(f"Dataset must contain more than K={k} observations.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(X_raw):,}")
c2.metric("Clustering features", len(FEATURES))
c3.metric("Selected K", k)
c4.metric("Dataset", source_name)
with st.expander("Preview data and feature summary"):
    st.dataframe(df.head(20), use_container_width=True)
    st.dataframe(feature_frame.describe().T, use_container_width=True)

st.subheader("2. Run experiment")
st.write(
    "The experiment compares Random, K-Means++, and PCA-score "
    "Quantile-Guided initialization under the same clustering and "
    "evaluation settings."
)
#run = st.button("▶ Run three-method experiment", type="primary")
run = st.button("▶ Run three-method experiment", type="primary", use_container_width=False)

if run:
    X = StandardScaler().fit_transform(X_raw) if standardize else X_raw.copy()
    
    # Select one shared silhouette sample for this experiment.
    silhouette_seed = int(base_seed)
    sil_rng = np.random.default_rng(silhouette_seed)

    silhouette_indices = np.sort(
        sil_rng.choice(
            len(X),
            size=min(int(silhouette_sample), len(X)),
            replace=False
        )
    )

    X_silhouette = X[silhouette_indices]
    methods = {
        "Random": lambda rng, seed: random_init(X, k, rng),
        "K-Means++": lambda rng, seed: kmeans_plus_plus_init(X, k, rng),
        "Quantile-Guided": lambda rng, seed: quantile_guided_init(X, k, seed),
    }
    records, anomalies = [], []
    progress = st.progress(0, text="Starting experiment…")
    total = trials * len(methods)
    completed = 0
    for trial in range(trials):
        seed = int(base_seed) + trial
        for name, init_fn in methods.items():
            started = time.perf_counter()
            try:
                rng = np.random.default_rng(seed)
                init_started = time.perf_counter()
                with PeakRSSSampler() as mem:
                    centers, init_info = init_fn(rng, seed)
                    init_seconds = time.perf_counter() - init_started
                    
                    cluster_started = time.perf_counter()

                    fit = lloyd_kmeans(
                        X, centers,
                        max_iter=int(max_iter),
                        tol=float(tol)
                    )

                    cluster_seconds = time.perf_counter() - cluster_started

                    # Bookkeeping after the timed clustering call
                    labels = np.asarray(fit["labels"], dtype=np.int64)

                    label_hash = hashlib.sha256(
                        labels.tobytes()
                    ).hexdigest()

                    partition_hash = canonical_partition_hash(labels)

                    # Evaluate the same observations for every method and trial
                    labels_sample = labels[silhouette_indices]

                    sil = calculate_silhouette(
                        X_silhouette,
                        labels_sample
                    )

                    if not np.isfinite(sil):
                        anomalies.append({
                            "method": name,
                            "trial": trial + 1,
                            "seed": seed,
                            "anomaly_type": "invalid_silhouette",
                            "details": (
                                "Silhouette score was NaN or infinite; "
                                "inspect the sampled cluster labels."
                            ),
                            "handling": (
                                "Recorded as undefined; exclude from mean "
                                "silhouette calculation and investigate."
                            )
                        })

                total_seconds = time.perf_counter() - started
                init_dist = int(init_info.get("initialization_distance_evaluations", 0))
                cluster_dist = int(fit["clustering_distance_evaluations"])
                events = fit["empty_cluster_events"]
                records.append({                                        
                    "method": name, "trial": trial + 1, "labels_sha256": label_hash, "partition_sha256": partition_hash, "seed": seed,
                    "dataset": source_name, "n_observations": len(X), "n_features": X.shape[1],
                    "K": k, "preprocessing": "StandardScaler" if standardize else "none",
                    "initialization_seconds": init_seconds, "clustering_seconds": cluster_seconds,
                    "total_seconds": total_seconds, "initialization_distance_evaluations": init_dist,
                    "clustering_distance_evaluations": cluster_dist,
                    "total_distance_evaluations": init_dist + cluster_dist,
                    "iterations": fit["iterations"], "converged": fit["converged"],
                    "peak_rss_bytes_sampled": mem.peak_bytes,
                    "peak_rss_mib_sampled": mem.peak_bytes / (1024 ** 2),
                    "sse_inertia": fit["inertia"], "silhouette_score": sil,
                    "initial_centers": json.dumps(np.asarray(centers).tolist()),
                    "initialization_diagnostics": json.dumps(init_info, default=str),
                    "empty_cluster_events": json.dumps(events),
                    "anomaly_count": len(events),
                    "status": "success",
                    "error": "",
                    "peak_rss_increase_bytes": mem.peak_increase_bytes,
                    "peak_rss_increase_mib": (
                        mem.peak_increase_bytes / (1024 ** 2)
                    ),
                })
                for event in events:
                    anomalies.append({"method": name, "trial": trial + 1, "seed": seed,
                                      "anomaly_type": "empty_cluster", "details": json.dumps(event),
                                      "handling": "previous centroid retained; inspect impact"})
                if not fit["converged"]:
                    anomalies.append({"method": name, "trial": trial + 1, "seed": seed,
                                      "anomaly_type": "non_convergence",
                                      "details": f"Reached max_iter={max_iter}",
                                      "handling": "recorded; interpret results with caution"})
            except Exception as exc:
                records.append({"method": name, "trial": trial + 1, "seed": seed,
                                "K": k, "status": "error", "error": repr(exc),
                                "total_seconds": time.perf_counter() - started})
                anomalies.append({"method": name, "trial": trial + 1, "seed": seed,
                                  "anomaly_type": "runtime_error", "details": repr(exc)})
            completed += 1
            progress.progress(completed / total, text=f"Completed {completed}/{total} runs")
    results = pd.DataFrame(records)
    anomaly_df = pd.DataFrame(anomalies)
    st.session_state["results"] = results
    st.session_state["anomalies"] = anomaly_df
    st.session_state["run_metadata"] = {
        "silhouette_sampling_seed": int(base_seed),
        "silhouette_sample_size_requested": int(silhouette_sample),
        "silhouette_sample_size_actual": len(silhouette_indices),
        "silhouette_sampling_method": (
            "Uniform sampling without replacement; selected once "
            "and reused across all methods and trials"
        ),
        "dataset": source_name, "observations": len(X), "features": FEATURES, "K": k,
        "trials": trials, "standardize": standardize, "max_iter": int(max_iter),
        "tolerance": float(tol), "silhouette_sample_size": int(silhouette_sample),
        "base_seed": int(base_seed),
        "distance_definition": "one point-to-centroid vector distance; clustering count includes final assignment pass",
        "memory_definition": "sampled peak process RSS, approximate",
        "quantile_guided_status": "implemented; validation pending",
        "quantile_guided_algorithm": "PCA-score quantile-ranked observed rows",
        "quantile_guided_rank_rule": "floor(p * (n - 1))",
        "quantile_guided_tie_rule": "ascending PC1 score; original row index breaks ties",
        "quantile_guided_duplicate_policy": "retain and report duplicate selected rows",
        "quantile_guided_initialization_distance_evaluations": 0
    }
    st.success("Baseline experiment finished. Review anomalies and compare metrics below.")

if "results" in st.session_state:
    results = st.session_state["results"]
    anomalies = st.session_state["anomalies"]
    successful = results[results["status"] == "success"].copy() if "status" in results else pd.DataFrame()
    st.subheader("3. Results dashboard")
    if successful.empty:
        st.warning("No successful runs to summarize. Check the error records.")
    else:
        summary = successful.groupby("method").agg(
            runs=("status", "count"),
            mean_time_s=("total_seconds", "mean"),
            mean_distances=("total_distance_evaluations", "mean"),
            mean_iterations=("iterations", "mean"),
            mean_peak_rss_mib=("peak_rss_mib_sampled", "mean"),
            mean_peak_rss_increase_mib=(
                "peak_rss_increase_mib", "mean"
            ),
            mean_sse=("sse_inertia", "mean"),
            mean_silhouette=("silhouette_score", "mean"),
            convergence_rate=("converged", "mean"),
        ).reset_index()
        cols = st.columns(4)
        cols[0].metric("Successful runs", f"{len(successful)}")
        cols[1].metric("Anomaly events", f"{len(anomalies)}")
        cols[2].metric("Methods run", f"{successful['method'].nunique()}")
        cols[3].metric("K", str(int(successful["K"].iloc[0])))
        st.dataframe(summary.round(4), use_container_width=True, hide_index=True)

        metric = st.selectbox("Chart metric", [
            "total_seconds", "total_distance_evaluations", "iterations",
            "peak_rss_mib_sampled", "sse_inertia", "silhouette_score"
        ], format_func=lambda x: {
            "total_seconds": "Total execution time (seconds)",
            "total_distance_evaluations": "Total distance evaluations",
            "iterations": "Iterations",
            "peak_rss_mib_sampled": "Sampled peak RSS (MiB)",
            "sse_inertia": "SSE / Inertia",
            "silhouette_score": "Silhouette Score"
        }[x])
        chart_data = successful.groupby("method")[metric].agg(["mean", "std"]).reset_index()
        st.bar_chart(chart_data.set_index("method")[["mean"]], use_container_width=True)
        st.caption("Bars show the mean across trials. Use the summary table and raw trial data to inspect variability; sampled RSS is approximate.")

        with st.expander("Per-trial results"):
            st.dataframe(results, use_container_width=True, hide_index=True)
        with st.expander("Anomalies and warnings"):
            if anomalies.empty:
                st.success("No anomalies were recorded.")
            else:
                st.dataframe(anomalies, use_container_width=True, hide_index=True)

        st.download_button("Download per-trial results (CSV)", successful.to_csv(index=False).encode("utf-8"),
                           "experiment_results.csv", "text/csv")
        st.download_button("Download summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                           "experiment_summary.csv", "text/csv")
        st.download_button("Download anomalies (CSV)", anomalies.to_csv(index=False).encode("utf-8"),
                           "experiment_anomalies.csv", "text/csv")
        st.download_button("Download run metadata (JSON)",
                           json.dumps(st.session_state["run_metadata"], indent=2).encode("utf-8"),
                           "run_metadata.json", "application/json")
else:
    st.subheader("3. Results dashboard")
    st.caption("Run the baseline experiment to populate the charts, summary table, anomaly log, and CSV downloads.")

st.divider()
st.caption(
    "Research integrity note: Quantile-Guided initialization is implemented; "
    "validate the implementation and experimental results before drawing "
    "comparative conclusions."
)
