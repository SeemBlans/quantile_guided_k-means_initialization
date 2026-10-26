# Quantile-Guided K-Means Research — VS Code starter

## Setup (Windows)
1. Open this folder in VS Code and copy `Final_Grades_Dataset_Cleaned.xlsx` into `data/`.
2. In the terminal run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Select `.venv` via `Ctrl+Shift+P` → `Python: Select Interpreter`.
4. Run `python main.py`.

The program implements Random and K-Means++ plus a transparent Lloyd K-Means loop. Quantile-Guided is deliberately pending: the multivariate rule must be approved before implementation. The program records that status rather than silently substituting an algorithm.

## Pilot configuration
K=3; 10 trials per method; four standardized grade features; maximum 300 Lloyd iterations; centroid-shift tolerance 1e-4; seeds 20261023 through 20261032. Silhouette uses Euclidean distance on standardized features and a reproducible sample of up to 5,000 records for runtime.

## Decide the multivariate Quantile-Guided rule

A four-feature row is a vector. Sorting each column independently and combining quantiles can destroy the original row-wise relationships. Select one of these options before implementation:

### Option A — Quantile-ranked observed rows
Define a fixed scalar score for each standardized four-feature row (the ranking rule must be specified and justified), sort by score while retaining original row indices, and select K complete rows at pre-specified quantile positions. This preserves observed records. Specify ties and duplicate selected rows.

### Option B — Synthetic marginal-quantile centroids
For each feature, compute quantiles at probabilities 0, 0.5, and 1 for K=3, then assemble each centroid from corresponding feature quantiles. These centroids may not exist as actual rows and can combine values from different records. Specify interpolation and duplicate-centroid reporting.

### Option C — PCA-score quantile-ranked observed rows
Standardize features, compute a deterministic first principal-component score, sort observations by that score, select K rows at pre-specified quantile positions, and use the original complete four-feature rows as centers. Record PCA fitting/projection as initialization overhead, selected original row indices, ties, and duplicate seeds.

For all options, quantile probabilities, quantile estimation, and integer row-index selection are distinct decisions. Linear interpolation estimates a quantile; it is not a rounding rule. For K=3, probabilities 0, 0.5, 1 are one possible convention, not a universal definition of Quantile-Guided K-Means.

## Metric definitions
- Distance evaluation = one point-to-centroid vector distance.
- Initialization distance evaluations are counted separately from clustering evaluations.
- Clustering count includes each Lloyd assignment pass and the final pass used for final labels/SSE.
- Total distance evaluations = initialization + clustering.
- Iterations count complete assignment/update rounds; the final labeling pass is not another iteration.
- SSE/Inertia is computed on standardized data.
- Silhouette uses Euclidean distance on the same standardized data.
- Memory is sampled peak process RSS and is approximate; short-lived peaks can be missed.
- Empty clusters are logged; current behavior retains the previous centroid.
- All methods use the same standardized feature matrix, K, convergence settings, and evaluation procedure.

## Exports
- `results/experiment_results_*.csv`: per-trial results
- `results/experiment_summary_*.csv`: mean and standard deviation for successful trials
- `logs/anomalies_*.csv`: anomalies and method-definition events
- `logs/run_metadata_*.json`: configuration and metric definitions

Keep the source workbook unchanged. The `#` column is excluded from clustering. Treat this as a pilot scaffold; do not claim superiority without completed, validated repeated trials.
