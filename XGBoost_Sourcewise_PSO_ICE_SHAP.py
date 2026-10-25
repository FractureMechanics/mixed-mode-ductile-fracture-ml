# -*- coding: utf-8 -*-


from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get( str(SCRIPT_DIR)))
OUTPUT_DIR = Path(os.environ.get( str(SCRIPT_DIR / "results_xgb_sourcewise")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fracture load(KN)"
GROUP_COL = "Source_ID"
ROW_ID_CANDIDATES = ["num.", "num", "sample_id", "Sample_ID", "ID"]
FEATURES = [
    "Toughness(MJ/m3)",
    "JIc(J/m2)",
    "thickness(mm)",
    "KI(pa.mm0.5)",
    "KII(pa.mm0.5)",
    "T-stress (Mpa)",
]

TEST_GROUP_FRACTION = 0.20
CV_FOLDS_REQUESTED = 5

PSO_POPULATION = int(os.environ.get("PSO_PARTICLES", "10"))
PSO_ITERATIONS = int(os.environ.get("PSO_ITERATIONS", "600"))
K_MAX = 600
W_INI, W_END, D1, D2 = 0.95, 0.40, 0.20, 7.0
C1_INI, C1_END = 2.5, 0.5
C2_INI, C2_END = 0.5, 2.5

N_ESTIMATORS_RANGE = (100, 1000)
MAX_DEPTH_RANGE = (3, 15)
LEARNING_RATE_RANGE = (0.01, 0.30)  

XGB_N_JOBS = -1
ICE_N_SAMPLES = 50
ICE_GRID_POINTS = 40
SHAP_BACKGROUND_N = 50
SHAP_EXPLAIN_N = 100
SHAP_MAX_EVALS = 100
SHOW_PLOTS = True

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE = "#00BCD4", "#1565C0"
COLOR_TEST_FACE, COLOR_TEST_EDGE = "#2ECC71", "#0B6623"
COLOR_OOD_FACE, COLOR_OOD_EDGE = "#D81B60", "black"


def normalize_source_id(value: Any) -> str:
    """Normalize DOI / publication identifiers for overlap checking."""
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = text.strip().rstrip("/.")
    return text


def load_and_validate(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put {path.name} in the same folder as this script, "
            "or set FRACTURE_DATA_DIR to the folder containing both Excel files."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    required = FEATURES + [TARGET, GROUP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing columns {missing}. Found: {df.columns.tolist()}")

    df = df.copy()
    df["_source_original"] = df[GROUP_COL].astype("string")
    df["_source_norm"] = df[GROUP_COL].map(normalize_source_id)
    if df["_source_norm"].eq("").any():
        bad_rows = df.index[df["_source_norm"].eq("")].tolist()[:10]
        raise ValueError(f"{dataset_name}: missing/blank Source_ID at row indices {bad_rows}.")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    numeric = df[FEATURES + [TARGET]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{dataset_name}: NaN or infinite numeric values detected in features/target.")

    if (df[TARGET] < 0).any():
        raise ValueError(f"{dataset_name}: negative fracture-load target values were found; verify data units.")

    return df.reset_index(drop=True)


def target_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in original target units; zero targets are excluded explicitly."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.abs(y_true) > np.finfo(float).eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "MAPE (%)": target_mape(y_true, y_pred),
        "RMSE (kN)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE (kN)": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_n": int(np.count_nonzero(np.abs(y_true) > np.finfo(float).eps)),
    }


def xgb_model_from_params(params: Dict[str, Any]) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=XGB_N_JOBS,
        tree_method="hist",
        **params,
    )


def decode_position(position: np.ndarray) -> Dict[str, Any]:
    """Map normalized PSO position [0,1]^3 onto the manuscript search space."""
    position = np.clip(np.asarray(position, dtype=float), 0.0, 1.0)
    n_estimators = int(round(N_ESTIMATORS_RANGE[0] + position[0] * (N_ESTIMATORS_RANGE[1] - N_ESTIMATORS_RANGE[0])))
    max_depth = int(round(MAX_DEPTH_RANGE[0] + position[1] * (MAX_DEPTH_RANGE[1] - MAX_DEPTH_RANGE[0])))
    log_lr_min = np.log(LEARNING_RATE_RANGE[0])
    log_lr_max = np.log(LEARNING_RATE_RANGE[1])
    learning_rate = float(np.exp(log_lr_min + position[2] * (log_lr_max - log_lr_min)))
    return {
        "n_estimators": int(np.clip(n_estimators, *N_ESTIMATORS_RANGE)),
        "max_depth": int(np.clip(max_depth, *MAX_DEPTH_RANGE)),
        "learning_rate": float(np.clip(learning_rate, *LEARNING_RATE_RANGE)),
    }


def inertia_weight(k: int) -> float:
    """Exponential decreasing inertia schedule as given by manuscript Eq. (9)."""
    # w_k = (w_ini - w_end - d1) * exp(k_max / (k_max + d2*k))
    return float((W_INI - W_END - D1) * np.exp(K_MAX / (K_MAX + D2 * k)))


def pso_coefficients(k: int) -> Tuple[float, float]:
    c1 = C1_INI - (C1_INI - C1_END) * (k / K_MAX)
    c2 = C2_INI + (C2_END - C2_INI) * (k / K_MAX)
    return float(c1), float(c2)


print("Loading datasets...")
df_dev = load_and_validate(CLEAN_FILE_PATH, "Development dataset")
df_ood = load_and_validate(UNSEEN_FILE_PATH, "OOD dataset")

print(f"Development: {len(df_dev)} observations, {df_dev['_source_norm'].nunique()} source groups")
print(f"OOD:         {len(df_ood)} observations, {df_ood['_source_norm'].nunique()} source groups")

source_overlap = set(df_dev["_source_norm"]) & set(df_ood["_source_norm"])
if source_overlap:
    raise ValueError(
        "Source_ID overlap detected between development and OOD datasets after normalization: "
        f"{sorted(source_overlap)}. Resolve source provenance before model evaluation."
    )

X_all = df_dev[FEATURES].copy()
y_all = df_dev[TARGET].copy()
groups_all = df_dev["_source_norm"].copy()

if groups_all.nunique() < 6:
    raise ValueError("At least 6 unique development sources are required for outer split plus 5-fold CV.")

outer_split = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRACTION, random_state=SEED)
train_idx, test_idx = next(outer_split.split(X_all, y_all, groups=groups_all))

train_df = df_dev.iloc[train_idx].copy().reset_index(drop=True)
test_df = df_dev.iloc[test_idx].copy().reset_index(drop=True)

train_sources = set(train_df["_source_norm"])
test_sources = set(test_df["_source_norm"])
assert train_sources.isdisjoint(test_sources), "Source leakage detected between outer training and test subsets."

print(f"Source-wise training set: {len(train_df)} rows from {len(train_sources)} sources")
print(f"Source-wise test set:     {len(test_df)} rows from {len(test_sources)} sources")

X_train_raw = train_df[FEATURES].copy()
y_train_raw = train_df[TARGET].copy()
groups_train = train_df["_source_norm"].copy()

n_splits = min(CV_FOLDS_REQUESTED, groups_train.nunique())
if n_splits < 2:
    raise ValueError("Too few source groups in training partition for GroupKFold.")
group_cv = GroupKFold(n_splits=n_splits)
cv_splits = list(group_cv.split(X_train_raw, y_train_raw, groups=groups_train))

# Record each training sample's validation-fold assignment and verify group disjointness.
cv_fold_for_train_row = np.full(len(train_df), -1, dtype=int)
cv_fold_rows = []
for fold_num, (fit_ids, val_ids) in enumerate(cv_splits, start=1):
    fit_sources = set(groups_train.iloc[fit_ids])
    val_sources = set(groups_train.iloc[val_ids])
    overlap = fit_sources & val_sources
    assert not overlap, f"Source leakage in fold {fold_num}: {sorted(overlap)}"
    cv_fold_for_train_row[val_ids] = fold_num
    cv_fold_rows.append({
        "Fold": fold_num,
        "Training_samples": int(len(fit_ids)),
        "Validation_samples": int(len(val_ids)),
        "Training_source_groups": int(len(fit_sources)),
        "Validation_source_groups": int(len(val_sources)),
        "Source_overlap_count": int(len(overlap)),
        "Validation_Source_IDs": "; ".join(sorted(val_sources)),
    })

if np.any(cv_fold_for_train_row < 1):
    raise RuntimeError("Some training observations were not assigned to a GroupKFold validation fold.")

objective_cache: Dict[Tuple[int, int, float], float] = {}
objective_evaluations = 0


def evaluate_candidate(params: Dict[str, Any]) -> float:
    """Mean grouped-CV MSE in the fold-specific Yeo-Johnson target space."""
    global objective_evaluations
    cache_key = (
        int(params["n_estimators"]),
        int(params["max_depth"]),
        round(float(params["learning_rate"]), 10),
    )
    if cache_key in objective_cache:
        return objective_cache[cache_key]

    fold_losses: List[float] = []
    for fit_ids, val_ids in cv_splits:
        X_fit = X_train_raw.iloc[fit_ids]
        X_val = X_train_raw.iloc[val_ids]
        y_fit = y_train_raw.iloc[fit_ids].to_numpy(dtype=float).reshape(-1, 1)
        y_val = y_train_raw.iloc[val_ids].to_numpy(dtype=float).reshape(-1, 1)

        x_transformer = PowerTransformer(method="yeo-johnson")
        y_transformer = PowerTransformer(method="yeo-johnson")
        X_fit_transformed = x_transformer.fit_transform(X_fit)
        X_val_transformed = x_transformer.transform(X_val)
        y_fit_transformed = y_transformer.fit_transform(y_fit).ravel()
        y_val_transformed = y_transformer.transform(y_val).ravel()

        model = xgb_model_from_params(params)
        model.fit(X_fit_transformed, y_fit_transformed, verbose=False)
        y_val_pred_transformed = model.predict(X_val_transformed)
        fold_losses.append(float(mean_squared_error(y_val_transformed, y_val_pred_transformed)))

    objective_evaluations += 1
    score = float(np.mean(fold_losses))
    if not np.isfinite(score):
        score = 1.0e30
    objective_cache[cache_key] = score
    return score


def run_pso() -> Tuple[Dict[str, Any], pd.DataFrame]:
    if PSO_POPULATION < 2 or PSO_ITERATIONS < 1:
        raise ValueError("PSO_POPULATION must be >=2 and PSO_ITERATIONS must be >=1.")

    rng = np.random.default_rng(SEED)
    dimensions = 3
    positions = rng.uniform(0.0, 1.0, size=(PSO_POPULATION, dimensions))
    velocities = rng.uniform(-0.1, 0.1, size=(PSO_POPULATION, dimensions))

    fitness = np.array([evaluate_candidate(decode_position(p)) for p in positions], dtype=float)
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()
    gbest_idx = int(np.argmin(pbest_fitness))
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = float(pbest_fitness[gbest_idx])

    history = [{
        "Iteration": 0,
        "Global_best_CV_MSE_transformed_target": gbest_fitness,
        "Iteration_best_CV_MSE_transformed_target": float(np.min(fitness)),
        "Mean_particle_CV_MSE_transformed_target": float(np.mean(fitness)),
        "Inertia_weight": inertia_weight(0),
        "c1": pso_coefficients(0)[0],
        "c2": pso_coefficients(0)[1],
        **decode_position(gbest_position),
    }]

    print(f"Initial best grouped-CV objective: {gbest_fitness:.6g}")
    for iteration in range(1, PSO_ITERATIONS + 1):
        k = min(iteration, K_MAX)
        w = inertia_weight(k)
        c1, c2 = pso_coefficients(k)
        r1 = rng.random(size=(PSO_POPULATION, dimensions))
        r2 = rng.random(size=(PSO_POPULATION, dimensions))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position.reshape(1, -1) - positions)
        )
        positions = np.clip(positions + velocities, 0.0, 1.0)
        fitness = np.array([evaluate_candidate(decode_position(p)) for p in positions], dtype=float)

        improved = fitness < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = fitness[improved]
        gbest_idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[gbest_idx] < gbest_fitness:
            gbest_fitness = float(pbest_fitness[gbest_idx])
            gbest_position = pbest_positions[gbest_idx].copy()

        history.append({
            "Iteration": iteration,
            "Global_best_CV_MSE_transformed_target": gbest_fitness,
            "Iteration_best_CV_MSE_transformed_target": float(np.min(fitness)),
            "Mean_particle_CV_MSE_transformed_target": float(np.mean(fitness)),
            "Inertia_weight": w,
            "c1": c1,
            "c2": c2,
            **decode_position(gbest_position),
        })

        if iteration == 1 or iteration % 10 == 0 or iteration == PSO_ITERATIONS:
            print(
                f"PSO {iteration:4d}/{PSO_ITERATIONS} | best CV MSE={gbest_fitness:.6g} | "
                f"params={decode_position(gbest_position)}"
            )

    best_params = decode_position(gbest_position)
    history_df = pd.DataFrame(history)
    return best_params, history_df


print("\nStarting source-wise PSO optimization...")
print(
    f"PSO configuration: population={PSO_POPULATION}, iterations={PSO_ITERATIONS}, "
    f"CV folds={n_splits}. This may take a long time for the full 600-iteration run."
)
best_params, pso_history_df = run_pso()
print("Best XGBoost parameters from grouped CV:", best_params)


X_scaler_final = PowerTransformer(method="yeo-johnson")
y_scaler_final = PowerTransformer(method="yeo-johnson")
X_train_transformed = X_scaler_final.fit_transform(X_train_raw)
y_train_transformed = y_scaler_final.fit_transform(y_train_raw.to_numpy(dtype=float).reshape(-1, 1)).ravel()

final_model = xgb_model_from_params(best_params)
final_model.fit(X_train_transformed, y_train_transformed, verbose=False)


def predict_raw(X_raw: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X_raw, pd.DataFrame):
        X_df = X_raw[FEATURES].copy()
    else:
        X_df = pd.DataFrame(np.asarray(X_raw), columns=FEATURES)
    X_transformed = X_scaler_final.transform(X_df)
    y_pred_transformed = final_model.predict(X_transformed).reshape(-1, 1)
    return y_scaler_final.inverse_transform(y_pred_transformed).ravel()


sets = {
    "Train": (train_df, X_train_raw, y_train_raw.to_numpy(dtype=float)),
    "Source-wise Test": (test_df, test_df[FEATURES].copy(), test_df[TARGET].to_numpy(dtype=float)),
    "OOD": (df_ood, df_ood[FEATURES].copy(), df_ood[TARGET].to_numpy(dtype=float)),
}

metric_rows = []
prediction_frames = []
prediction_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
for set_name, (source_df, X_set, y_set) in sets.items():
    y_pred = predict_raw(X_set)
    prediction_cache[set_name] = (y_set, y_pred)
    metrics = calculate_metrics(y_set, y_pred)
    source_col = source_df["_source_original"].astype(str).to_numpy()
    metric_rows.append({
        "Dataset": set_name,
        "N": int(len(y_set)),
        "Source_groups": int(source_df["_source_norm"].nunique()),
        **metrics,
    })

    row_ids = None
    for candidate in ROW_ID_CANDIDATES:
        if candidate in source_df.columns:
            row_ids = source_df[candidate].astype(str).to_numpy()
            break
    if row_ids is None:
        row_ids = source_df.index.astype(str).to_numpy()

    prediction_frames.append(pd.DataFrame({
        "Dataset": set_name,
        "Sample_ID": row_ids,
        "Source_ID": source_col,
        "Actual fracture load (kN)": y_set,
        "Predicted fracture load (kN)": y_pred,
        "Residual (Predicted - Actual) (kN)": y_pred - y_set,
        "Absolute error (kN)": np.abs(y_pred - y_set),
    }))

metrics_df = pd.DataFrame(metric_rows)
predictions_df = pd.concat(prediction_frames, ignore_index=True)

print("\nFinal performance metrics (all samples included):")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

partition = pd.DataFrame({
    "Development_row_index": np.arange(len(df_dev)),
    "Sample_ID": [str(df_dev.iloc[i][c]) if (c := next((x for x in ROW_ID_CANDIDATES if x in df_dev.columns), None)) else str(i) for i in range(len(df_dev))],
    "Source_ID": df_dev["_source_original"].astype(str).to_numpy(),
    "Source_ID_normalized": df_dev["_source_norm"].to_numpy(),
    "Split": "",
    "CV_validation_fold": pd.Series([pd.NA] * len(df_dev), dtype="Int64"),
})
partition.loc[train_idx, "Split"] = "Train"
partition.loc[test_idx, "Split"] = "Source-wise Test"
for local_i, original_i in enumerate(train_idx):
    partition.loc[original_i, "CV_validation_fold"] = int(cv_fold_for_train_row[local_i])

ood_partition = pd.DataFrame({
    "OOD_row_index": np.arange(len(df_ood)),
    "Sample_ID": [str(df_ood.iloc[i][c]) if (c := next((x for x in ROW_ID_CANDIDATES if x in df_ood.columns), None)) else str(i) for i in range(len(df_ood))],
    "Source_ID": df_ood["_source_original"].astype(str).to_numpy(),
    "Source_ID_normalized": df_ood["_source_norm"].to_numpy(),
    "Split": "OOD",
})

fold_summary_df = pd.DataFrame(cv_fold_rows)
source_count_df = (
    pd.concat([
        partition[["Source_ID_normalized", "Split"]].rename(columns={"Source_ID_normalized": "Source_ID"}),
        ood_partition[["Source_ID_normalized", "Split"]].rename(columns={"Source_ID_normalized": "Source_ID"}),
    ], ignore_index=True)
    .groupby(["Split", "Source_ID"], dropna=False)
    .size().reset_index(name="N_samples")
)

def plot_scatter_panels(zoom: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    plot_specs = [
        ("Train", COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE),
        ("Source-wise Test", COLOR_TEST_FACE, COLOR_TEST_EDGE),
        ("OOD", COLOR_OOD_FACE, COLOR_OOD_EDGE),
    ]

    if zoom:
        axis_max = 60.0
        fig_title = "Actual vs. predicted fracture loads (visual zoom ≤60 kN; metrics use all observations)"
        output_path = OUTPUT_DIR / "XGBoost_Sourcewise_Scatter_Zoom_0_60kN.png"
    else:
        all_values = np.concatenate([np.concatenate(prediction_cache[name]) for name, _, _ in plot_specs])
        axis_max = max(1.0, float(np.nanmax(all_values)) * 1.03)
        fig_title = "Actual vs. predicted fracture loads (all observations)"
        output_path = OUTPUT_DIR / "XGBoost_Sourcewise_Scatter_AllData.png"

    for ax, (set_name, face_color, edge_color) in zip(axes, plot_specs):
        y_true, y_pred = prediction_cache[set_name]
        visible = np.ones(len(y_true), dtype=bool)
        if zoom:
            visible = (y_true <= 60.0) & (y_pred <= 60.0)

        ax.scatter(
            y_true[visible], y_pred[visible],
            facecolors=face_color, edgecolors=edge_color,
            s=65, alpha=0.85, linewidths=1.0, label="Observations", zorder=3
        )
        ax.plot([0, axis_max], [0, axis_max], "k--", lw=1.8, label="Ideal line", zorder=2)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(set_name, fontweight="bold")
        ax.set_xlabel("Actual fracture load (kN)")
        ax.set_ylabel("Predicted fracture load (kN)")
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=False)
        metric = next(row for row in metric_rows if row["Dataset"] == set_name)
        ax.text(
            0.04, 0.96,
            f"$R^2={metric['R2']:.3f}$\nMAPE={metric['MAPE (%)']:.2f}%\nN={metric['N']}",
            transform=ax.transAxes, va="top", ha="left", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, boxstyle="round,pad=0.3")
        )

    fig.suptitle(fig_title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


plot_scatter_panels(zoom=False)
plot_scatter_panels(zoom=True)

print("\nCreating ICE plot for T-stress...")
ICE_FEATURE = "T-stress (Mpa)"
rng = np.random.default_rng(SEED)
num_ice = min(ICE_N_SAMPLES, len(X_train_raw))
X_ice = X_train_raw.sample(n=num_ice, random_state=SEED).copy()
feature_low, feature_high = np.nanpercentile(X_train_raw[ICE_FEATURE], [5, 95])
ice_grid = np.linspace(feature_low, feature_high, ICE_GRID_POINTS)
ice_curves = np.zeros((num_ice, len(ice_grid)), dtype=float)

for sample_i, (_, sample) in enumerate(X_ice.iterrows()):
    grid_df = pd.concat([sample.to_frame().T] * len(ice_grid), ignore_index=True)
    grid_df[ICE_FEATURE] = ice_grid
    grid_df = grid_df[FEATURES].astype(float)
    ice_curves[sample_i, :] = predict_raw(grid_df)

fig, ax = plt.subplots(figsize=(8, 6.5))
for curve in ice_curves:
    ax.plot(ice_grid, curve, color="#4169E1", alpha=0.20, linewidth=0.8)
ax.plot(ice_grid, np.mean(ice_curves, axis=0), color="#D81B60", linestyle="--", linewidth=3.0, label="Average ICE")
ax.set_title("ICE: Influence of T-stress on predicted fracture load", fontweight="bold")
ax.set_xlabel(ICE_FEATURE)
ax.set_ylabel("Predicted fracture load (kN)")
ax.grid(True, linestyle=":", alpha=0.35)
ax.legend(frameon=True)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "XGBoost_ICE_Tstress.png", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

print("\nStarting SHAP permutation analysis...")
shap_importance_df = pd.DataFrame(columns=["Feature", "Mean_abs_SHAP"])
shap_error_path = OUTPUT_DIR / "SHAP_error.txt"

try:
    background = X_train_raw.sample(n=min(SHAP_BACKGROUND_N, len(X_train_raw)), random_state=SEED)
    explain_df = X_train_raw.sample(n=min(SHAP_EXPLAIN_N, len(X_train_raw)), random_state=SEED + 1)

    def model_fn_for_shap(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        pred = predict_raw(pd.DataFrame(arr, columns=FEATURES))
        return np.asarray(pred, dtype=float).ravel()

    explainer = shap.PermutationExplainer(model_fn_for_shap, background.to_numpy(dtype=float))
    max_evals = max(int(SHAP_MAX_EVALS), 2 * len(FEATURES) + 1)
    shap_values = explainer(explain_df.to_numpy(dtype=float), max_evals=max_evals)
    shap_array = np.asarray(shap_values.values)

    shap_importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Mean_abs_SHAP": np.mean(np.abs(shap_array), axis=0),
    }).sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_array, explain_df, show=False, plot_size=(10, 7))
    plt.title("SHAP global interpretation (permutation explainer)", fontsize=15, fontweight="bold", pad=18)
    plt.xlabel("SHAP value (impact on predicted fracture load, kN)", fontsize=12, fontweight="bold")
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "XGBoost_SHAP_Summary.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    if shap_error_path.exists():
        shap_error_path.unlink()

except Exception as exc:
    message = f"SHAP analysis failed: {type(exc).__name__}: {exc}\n"
    shap_error_path.write_text(message, encoding="utf-8")
    print(message.strip())
    print("The model metrics and other outputs were still saved; check SHAP_error.txt for details.")

joblib.dump(
    {
        "model": final_model,
        "x_scaler": X_scaler_final,
        "y_scaler": y_scaler_final,
        "features": FEATURES,
        "target": TARGET,
        "best_params": best_params,
        "seed": SEED,
        "sourcewise_train_source_ids": sorted(train_sources),
        "sourcewise_test_source_ids": sorted(test_sources),
        "ood_source_ids": sorted(set(df_ood["_source_norm"])),
    },
    OUTPUT_DIR / "XGBoost_Sourcewise_ModelBundle.joblib",
)

config = {
    "seed": SEED,
    "data_dir": str(DATA_DIR),
    "development_file": CLEAN_FILE_PATH.name,
    "ood_file": UNSEEN_FILE_PATH.name,
    "features": FEATURES,
    "target": TARGET,
    "grouping_variable": GROUP_COL,
    "outer_split": "GroupShuffleSplit",
    "outer_test_group_fraction": TEST_GROUP_FRACTION,
    "cv": f"GroupKFold(n_splits={n_splits})",
    "pso_population": PSO_POPULATION,
    "pso_iterations": PSO_ITERATIONS,
    "pso_schedule": {
        "w_ini": W_INI, "w_end": W_END, "d1": D1, "d2": D2, "k_max": K_MAX,
        "c1_ini": C1_INI, "c1_end": C1_END, "c2_ini": C2_INI, "c2_end": C2_END,
    },
    "search_ranges": {
        "n_estimators": list(N_ESTIMATORS_RANGE),
        "max_depth": list(MAX_DEPTH_RANGE),
        "learning_rate_log_uniform": list(LEARNING_RATE_RANGE),
    },
    "best_params": best_params,
    "note": "All observations included; no fracture-load cutoff; OOD never used for training or tuning.",
}
(OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
