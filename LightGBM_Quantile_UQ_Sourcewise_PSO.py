# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_pinball_loss,
)

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FRACTURE_DATA_DIR", str(SCRIPT_DIR)))
OUTPUT_DIR = Path(os.environ.get(
    "FRACTURE_OUTPUT_DIR", str(SCRIPT_DIR / "results_lgbm_quantile_sourcewise")
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "fracture load(KN)"
GROUP_COL = "Source_ID"
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

PSO_POPULATION = int(os.environ.get("LGBM_PSO_PARTICLES", "10"))
PSO_ITERATIONS = int(os.environ.get("LGBM_PSO_ITERATIONS", "600"))
K_MAX = 600
W_INI, W_END, D1, D2 = 0.95, 0.40, 0.20, 7.0
C1_INI, C1_END = 2.5, 0.5
C2_INI, C2_END = 0.5, 2.5

NUM_LEAVES_RANGE = (10, 100)
MIN_CHILD_SAMPLES_RANGE = (5, 50)
LEARNING_RATE_RANGE = (0.01, 0.30)  
N_ESTIMATORS_FIXED = 100  

QUANTILES = (0.025, 0.50, 0.975)
LGBM_N_JOBS = int(os.environ.get("LGBM_N_JOBS", "-1"))
SHOW_PLOTS = os.environ.get("LGBM_SHOW_PLOTS", "1") == "1"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE = "#00BCD4", "#1565C0"
COLOR_TEST_FACE, COLOR_TEST_EDGE = "#2ECC71", "#0B6623"
COLOR_OOD_FACE, COLOR_OOD_EDGE = "#D81B60", "black"
COLOR_INTERVAL = "#90A4AE"


def normalize_source_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().casefold()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi\s*:\s*", "", value)
    return value.strip().rstrip("/.")


def load_and_validate(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put {path.name} next to this script, or set FRACTURE_DATA_DIR."
        )
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    required = FEATURES + [TARGET, GROUP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing columns {missing}; available={df.columns.tolist()}")

    df = df.copy()
    df["_source_original"] = df[GROUP_COL].astype("string")
    df["_source_norm"] = df[GROUP_COL].map(normalize_source_id)
    if df["_source_norm"].eq("").any():
        rows = df.index[df["_source_norm"].eq("")].tolist()[:10]
        raise ValueError(f"{dataset_name}: empty Source_ID at rows {rows}.")
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    numeric = df[FEATURES + [TARGET]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{dataset_name}: NaN or infinite values in features/target.")
    if (df[TARGET] < 0).any():
        raise ValueError(f"{dataset_name}: negative target values found; verify units/data.")
    return df.reset_index(drop=True)


def target_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.abs(y_true) > np.finfo(float).eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "R2 (median)": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "MAPE (%) (median)": target_mape(y_true, y_pred),
        "RMSE (kN) (median)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE (kN) (median)": float(mean_absolute_error(y_true, y_pred)),
        "N": int(len(y_true)),
        "MAPE denominator N": int(np.count_nonzero(np.abs(y_true) > np.finfo(float).eps)),
    }


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    raw_quantiles: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Dict[str, float]:
    """Evaluate ordered intervals; compute pinball loss on raw quantile predictions."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    median = np.asarray(median, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    if raw_quantiles is None:
        raw_lower, raw_median, raw_upper = lower, median, upper
    else:
        raw_lower, raw_median, raw_upper = [np.asarray(v, dtype=float).ravel() for v in raw_quantiles]
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    alpha = 0.05
    winkler = widths + (2.0 / alpha) * np.maximum(lower - y_true, 0) + (2.0 / alpha) * np.maximum(y_true - upper, 0)
    return {
        "Empirical interval coverage (%)": float(np.mean(covered) * 100),
        "Mean interval width (kN)": float(np.mean(widths)),
        "Median interval width (kN)": float(np.median(widths)),
        "Mean 95% interval score (kN)": float(np.mean(winkler)),
        "Pinball loss q=0.025 (raw quantile model)": float(mean_pinball_loss(y_true, raw_lower, alpha=0.025)),
        "Pinball loss q=0.50 (raw quantile model)": float(mean_pinball_loss(y_true, raw_median, alpha=0.50)),
        "Pinball loss q=0.975 (raw quantile model)": float(mean_pinball_loss(y_true, raw_upper, alpha=0.975)),
    }


def lgbm_point_model(params: Dict[str, Any]) -> lgb.LGBMRegressor:
    """LightGBM point regressor used as the PSO fitness model."""
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS_FIXED,
        random_state=SEED,
        n_jobs=LGBM_N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **params,
    )


def lgbm_quantile_model(params: Dict[str, Any], alpha: float) -> lgb.LGBMRegressor:
    """LightGBM quantile regressor for the requested conditional quantile."""
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=N_ESTIMATORS_FIXED,
        random_state=SEED,
        n_jobs=LGBM_N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **params,
    )


def decode_position(position: np.ndarray) -> Dict[str, Any]:
    """Map normalized PSO position in [0,1]^3 to manuscript LightGBM ranges."""
    position = np.clip(np.asarray(position, dtype=float), 0.0, 1.0)
    leaves = int(round(NUM_LEAVES_RANGE[0] + position[0] * (NUM_LEAVES_RANGE[1] - NUM_LEAVES_RANGE[0])))
    min_child_samples = int(round(
        MIN_CHILD_SAMPLES_RANGE[0]
        + position[1] * (MIN_CHILD_SAMPLES_RANGE[1] - MIN_CHILD_SAMPLES_RANGE[0])
    ))
    log_lr_low = np.log(LEARNING_RATE_RANGE[0])
    log_lr_high = np.log(LEARNING_RATE_RANGE[1])
    learning_rate = float(np.exp(log_lr_low + position[2] * (log_lr_high - log_lr_low)))
    return {
        "num_leaves": int(np.clip(leaves, *NUM_LEAVES_RANGE)),
        "min_child_samples": int(np.clip(min_child_samples, *MIN_CHILD_SAMPLES_RANGE)),
        "learning_rate": float(np.clip(learning_rate, *LEARNING_RATE_RANGE)),
    }


def inertia_weight(k: int) -> float:
    return float((W_INI - W_END - D1) * np.exp(K_MAX / (K_MAX + D2 * k)))


def pso_coefficients(k: int) -> Tuple[float, float]:
    c1 = C1_INI - (C1_INI - C1_END) * (k / K_MAX)
    c2 = C2_INI + (C2_END - C2_INI) * (k / K_MAX)
    return float(c1), float(c2)


def fit_fold_transformers(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame):
    x_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    y_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    Xtr = x_transformer.fit_transform(X_train)
    Xva = x_transformer.transform(X_valid)
    ytr = y_transformer.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
    return Xtr, ytr, Xva, x_transformer, y_transformer


print("Loading development and OOD datasets...")
if not hasattr(lgb, "LGBMRegressor"):
    raise RuntimeError("A working LightGBM installation is required.")

df_dev = load_and_validate(CLEAN_FILE_PATH, "Development dataset")
df_ood = load_and_validate(UNSEEN_FILE_PATH, "OOD dataset")
print(f"Development: {len(df_dev)} rows, {df_dev['_source_norm'].nunique()} sources")
print(f"OOD:         {len(df_ood)} rows, {df_ood['_source_norm'].nunique()} sources")
source_overlap = set(df_dev["_source_norm"]) & set(df_ood["_source_norm"])
if source_overlap:
    raise ValueError(f"Development/OOD publication-source overlap detected: {sorted(source_overlap)}")
print("Publication-level Source_ID overlap: 0")

if df_dev["_source_norm"].nunique() < 6:
    raise ValueError("Need at least 6 development sources for the outer source split and 5-fold GroupKFold.")

outer = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRACTION, random_state=SEED)
train_idx, test_idx = next(outer.split(df_dev[FEATURES], df_dev[TARGET], groups=df_dev["_source_norm"]))
train_df = df_dev.iloc[train_idx].reset_index(drop=True)
test_df = df_dev.iloc[test_idx].reset_index(drop=True)
train_sources = set(train_df["_source_norm"])
test_sources = set(test_df["_source_norm"])
assert train_sources.isdisjoint(test_sources), "Source overlap between train and test."
print(f"Training set: {len(train_df)} rows, {len(train_sources)} sources")
print(f"Source-wise test set: {len(test_df)} rows, {len(test_sources)} sources")

X_train_raw = train_df[FEATURES].copy()
y_train_raw = train_df[TARGET].copy()
groups_train = train_df["_source_norm"].copy()
n_splits = min(CV_FOLDS_REQUESTED, int(groups_train.nunique()))
if n_splits < 2:
    raise ValueError("Not enough unique sources in outer training subset for GroupKFold.")
gkf = GroupKFold(n_splits=n_splits)
cv_splits = list(gkf.split(X_train_raw, y_train_raw, groups=groups_train))
for fold_no, (itr, iva) in enumerate(cv_splits, start=1):
    a = set(groups_train.iloc[itr])
    b = set(groups_train.iloc[iva])
    assert a.isdisjoint(b), f"Source overlap inside CV fold {fold_no}."
print(f"GroupKFold: {n_splits} folds; no publication-source overlap across each fold.")


objective_cache: Dict[Tuple[int, int, float], float] = {}
cv_details: List[Dict[str, Any]] = []


def objective_position(position: np.ndarray) -> float:
    params = decode_position(position)
    key = (params["num_leaves"], params["min_child_samples"], round(params["learning_rate"], 10))
    if key in objective_cache:
        return objective_cache[key]
    fold_mses = []
    for fold_no, (itr, iva) in enumerate(cv_splits, start=1):
        Xtr_raw, Xva_raw = X_train_raw.iloc[itr], X_train_raw.iloc[iva]
        ytr_raw, yva_raw = y_train_raw.iloc[itr], y_train_raw.iloc[iva]
        Xtr, ytr_s, Xva, x_scaler_fold, y_scaler_fold = fit_fold_transformers(Xtr_raw, ytr_raw, Xva_raw)
        model = lgbm_point_model(params)
        model.fit(Xtr, ytr_s)
        pred_s = np.asarray(model.predict(Xva)).reshape(-1, 1)
        pred = y_scaler_fold.inverse_transform(pred_s).ravel()
        mse = float(mean_squared_error(np.asarray(yva_raw), pred))
        fold_mses.append(mse)
        cv_details.append({
            "evaluation_type": "PSO_objective_fold",
            "params_key": str(key),
            "fold": fold_no,
            "n_train": len(itr),
            "n_valid": len(iva),
            "n_valid_sources": int(groups_train.iloc[iva].nunique()),
            "MSE_kN2": mse,
        })
    loss = float(np.mean(fold_mses))
    objective_cache[key] = loss
    return loss


def run_pso() -> Tuple[Dict[str, Any], pd.DataFrame, float]:
    dim = 3
    rng = np.random.default_rng(SEED)
    positions = rng.uniform(0.0, 1.0, size=(PSO_POPULATION, dim))
    velocities = rng.uniform(-0.1, 0.1, size=(PSO_POPULATION, dim))
    personal_best_positions = positions.copy()
    personal_best_losses = np.full(PSO_POPULATION, np.inf, dtype=float)
    history: List[Dict[str, Any]] = []
    global_best_position = positions[0].copy()
    global_best_loss = np.inf

    print(f"Starting custom PSO: {PSO_POPULATION} particles x {PSO_ITERATIONS} iterations; CV folds={n_splits}.")
    for i in range(PSO_POPULATION):
        loss = objective_position(positions[i])
        personal_best_losses[i] = loss
        if loss < global_best_loss:
            global_best_loss = loss
            global_best_position = positions[i].copy()
        p = decode_position(positions[i])
        history.append({"iteration": 0, "particle": i, **p, "CV_MSE_kN2": loss, "best_so_far_MSE_kN2": global_best_loss})

    for k in range(1, PSO_ITERATIONS + 1):
        w = inertia_weight(k)
        c1, c2 = pso_coefficients(k)
        for i in range(PSO_POPULATION):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - positions[i])
                + c2 * r2 * (global_best_position - positions[i])
            )
            velocities[i] = np.clip(velocities[i], -0.25, 0.25)
            positions[i] = np.clip(positions[i] + velocities[i], 0.0, 1.0)
            loss = objective_position(positions[i])
            if loss < personal_best_losses[i]:
                personal_best_losses[i] = loss
                personal_best_positions[i] = positions[i].copy()
            if loss < global_best_loss:
                global_best_loss = loss
                global_best_position = positions[i].copy()
            p = decode_position(positions[i])
            history.append({
                "iteration": k,
                "particle": i,
                **p,
                "CV_MSE_kN2": loss,
                "best_so_far_MSE_kN2": global_best_loss,
                "inertia_weight": w,
                "c1": c1,
                "c2": c2,
            })
        if k == 1 or k % max(1, PSO_ITERATIONS // 20) == 0 or k == PSO_ITERATIONS:
            print(f"PSO iteration {k}/{PSO_ITERATIONS}: best CV MSE={global_best_loss:.6g} kN^2")

    return decode_position(global_best_position), pd.DataFrame(history), float(global_best_loss)


best_params, pso_history, best_cv_mse = run_pso()
print("Best PSO parameters:", best_params)
print(f"Best source-wise CV MSE (kN^2): {best_cv_mse:.6f}")

iter_best = pso_history.groupby("iteration", as_index=False)["best_so_far_MSE_kN2"].min()
fig, ax = plt.subplots(figsize=(8.5, 5.2))
per_iter_obs = pso_history.groupby("iteration", as_index=False)["CV_MSE_kN2"].min()
ax.scatter(per_iter_obs["iteration"], per_iter_obs["CV_MSE_kN2"], color="red", alpha=0.65, s=38, label="Best particle loss per iteration", zorder=2)
ax.plot(iter_best["iteration"], iter_best["best_so_far_MSE_kN2"], color="#1565C0", lw=2.5, marker="o", markersize=4, label="Global best CV MSE", zorder=3)
ax.set_title("LightGBM Quantile-UQ Hyperparameter Optimization", fontweight="bold")
ax.set_xlabel("PSO iteration")
ax.set_ylabel("Source-wise CV MSE (kN$^2$)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, edgecolor="black", fancybox=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pso_convergence.png", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

X_test_scaled = x_scaler.transform(test_df[FEATURES])
X_ood_scaled = x_scaler.transform(df_ood[FEATURES])
y_test_true = test_df[TARGET].to_numpy(dtype=float)
y_ood_true = df_ood[TARGET].to_numpy(dtype=float)
y_train_true = y_train_raw.to_numpy(dtype=float)

quantile_models: Dict[float, lgb.LGBMRegressor] = {}
for q in QUANTILES:
    q_model = lgbm_quantile_model(best_params, q)
    q_model.fit(X_train_scaled, y_train_scaled)
    quantile_models[float(q)] = q_model


def predict_quantiles(X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    raw_scaled = np.column_stack([
        quantile_models[q].predict(X_scaled).ravel() for q in QUANTILES
    ])
    raw_original = np.column_stack([
        y_scaler.inverse_transform(raw_scaled[:, j].reshape(-1, 1)).ravel()
        for j in range(raw_scaled.shape[1])
    ])
    crossing = (raw_original[:, 0] > raw_original[:, 1]) | (raw_original[:, 1] > raw_original[:, 2])
    ordered = np.sort(raw_original, axis=1)
    lower, median, upper = ordered[:, 0], ordered[:, 1], ordered[:, 2]
    return (
        lower, median, upper, float(np.mean(crossing) * 100.0),
        raw_original[:, 0], raw_original[:, 1], raw_original[:, 2],
    )


preds = {}
train_X_scaled = X_train_scaled
for name, X_scaled, y_true in [
    ("Train", train_X_scaled, y_train_true),
    ("Sourcewise_Test", X_test_scaled, y_test_true),
    ("OOD", X_ood_scaled, y_ood_true),
]:
    lower, median, upper, crossing_rate, raw_lower, raw_median, raw_upper = predict_quantiles(X_scaled)
    metrics = regression_metrics(y_true, median)
    uncertainty = interval_metrics(
        y_true, lower, median, upper, raw_quantiles=(raw_lower, raw_median, raw_upper)
    )
    metrics.update(uncertainty)
    metrics["Quantile crossing rate before rearrangement (%)"] = crossing_rate
    preds[name] = {
        "y_true": y_true,
        "lower": lower,
        "median": median,
        "upper": upper,
        "metrics": metrics,
    }
    print(
        f"{name}: R2={metrics['R2 (median)']:.4f}, "
        f"MAPE={metrics['MAPE (%) (median)']:.3f}%, "
        f"95% interval coverage={metrics['Empirical interval coverage (%)']:.2f}%"
    )

all_values = []
for p in preds.values():
    all_values.extend(p["y_true"].tolist())
    all_values.extend(p["lower"].tolist())
    all_values.extend(p["upper"].tolist())
finite_values = np.asarray(all_values, dtype=float)
finite_values = finite_values[np.isfinite(finite_values)]
plot_min = min(0.0, float(np.min(finite_values)))
plot_max = max(1.0, float(np.max(finite_values)))
pad = 0.04 * (plot_max - plot_min)
plot_min -= pad
plot_max += pad

plot_specs = [
    ("Train", "Training Data", COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE),
    ("Sourcewise_Test", "Source-wise Test Data", COLOR_TEST_FACE, COLOR_TEST_EDGE),
    ("OOD", "Independent OOD Data", COLOR_OOD_FACE, COLOR_OOD_EDGE),
]
fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))
for ax, (key, title, face, edge) in zip(axes, plot_specs):
    p = preds[key]
    yt, lo, med, up = p["y_true"], p["lower"], p["median"], p["upper"]
    # Show every sample; no error-based cleaning or target truncation.
    ax.vlines(yt, lo, up, color=COLOR_INTERVAL, alpha=0.24, linewidth=0.9, label="95% quantile interval", zorder=1)
    ax.scatter(yt, med, facecolors=face, edgecolors=edge, s=65, alpha=0.88, linewidths=1.0, label="Median prediction", zorder=4)
    ax.plot([plot_min, plot_max], [plot_min, plot_max], "k--", lw=1.8, label="Ideal line", zorder=3)
    m = p["metrics"]
    stats_text = (
        f"$R^2={m['R2 (median)']:.3f}$\n"
        f"MAPE={m['MAPE (%) (median)']:.2f}%\n"
        f"Coverage={m['Empirical interval coverage (%)']:.1f}%\n"
        f"$Width={m['Mean interval width (kN)']:.2f}$ kN"
    )
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, va="top", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, boxstyle="round,pad=0.35"))
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Load (kN)")
    ax.set_ylabel("Predicted Load (kN)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=False, fontsize=9)
fig.tight_layout(pad=1.8)
fig.savefig(OUTPUT_DIR / "actual_vs_predicted_quantile_uq.png", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for key, label, color in [
    ("Train", "Train", COLOR_TRAIN_EDGE),
    ("Sourcewise_Test", "Source-wise Test", COLOR_TEST_EDGE),
    ("OOD", "OOD", COLOR_OOD_EDGE),
]:
    p = preds[key]
    residual = p["y_true"] - p["median"]
    ax.scatter(p["median"], residual, s=42, alpha=0.75, edgecolors="black", linewidths=0.4, label=label)
ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Median Predicted Load (kN)")
ax.set_ylabel("Residual (Actual - Predicted, kN)")
ax.set_title("Residual Plot: LightGBM Quantile Regression", fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.45)
ax.legend(frameon=True, edgecolor="black", fancybox=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "residuals_quantile_uq.png", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)


summary_rows = []
for dataset_name in ["Train", "Sourcewise_Test", "OOD"]:
    summary_rows.append({"Dataset": dataset_name, **preds[dataset_name]["metrics"]})
summary_df = pd.DataFrame(summary_rows)

prediction_frames = []
for dataset_name, df_part in [
    ("Train", train_df),
    ("Sourcewise_Test", test_df),
    ("OOD", df_ood),
]:
    p = preds[dataset_name]
    frame = df_part.copy()
    frame.insert(0, "Evaluation_Set", dataset_name)
    frame["Predicted_q025_kN"] = p["lower"]
    frame["Predicted_median_kN"] = p["median"]
    frame["Predicted_q975_kN"] = p["upper"]
    frame["Interval_Width_kN"] = p["upper"] - p["lower"]
    frame["Covered_by_95pct_interval"] = (p["y_true"] >= p["lower"]) & (p["y_true"] <= p["upper"])
    frame["Median_Residual_kN"] = p["y_true"] - p["median"]
    prediction_frames.append(frame)
predictions_df = pd.concat(prediction_frames, ignore_index=True)

split_audit_rows = []
for part_name, dff in [("Train", train_df), ("Sourcewise_Test", test_df), ("OOD", df_ood)]:
    for source, count in dff.groupby("_source_norm").size().items():
        split_audit_rows.append({
            "Dataset": part_name,
            "Source_ID_normalized": source,
            "N_rows": int(count),
            "N_sources_in_dataset": int(dff["_source_norm"].nunique()),
        })
split_audit_df = pd.DataFrame(split_audit_rows)
source_overlap_df = pd.DataFrame({
    "Development_source_ID_normalized": pd.Series(sorted(set(df_dev["_source_norm"]))),
    "OOD_source_ID_normalized": pd.Series(sorted(set(df_ood["_source_norm"]))),
})
params_df = pd.DataFrame([{
    **best_params,
    "best_sourcewise_CV_MSE_kN2": best_cv_mse,
    "PSO_particles": PSO_POPULATION,
    "PSO_iterations": PSO_ITERATIONS,
    "CV_folds": n_splits,
    "outer_test_source_fraction": TEST_GROUP_FRACTION,
    "quantiles": str(QUANTILES),
    "LightGBM_version": lgb.__version__,
}])

config = {
    "seed": SEED,
    "development_file": str(CLEAN_FILE_PATH),
    "ood_file": str(UNSEEN_FILE_PATH),
    "features": FEATURES,
    "target": TARGET,
    "group_column": GROUP_COL,
    "outer_split": "GroupShuffleSplit by normalized Source_ID; test_size=0.20; random_state=42",
    "inner_cv": f"GroupKFold(n_splits={n_splits}) by normalized Source_ID",
    "preprocessing": "PowerTransformer(method='yeo-johnson') fitted inside each CV training fold; final scaler fitted only on outer training partition",
    "pso": {
        "particles": PSO_POPULATION,
        "iterations": PSO_ITERATIONS,
        "K_MAX": K_MAX,
        "W_INI": W_INI,
        "W_END": W_END,
        "D1": D1,
        "D2": D2,
        "c1_initial": C1_INI,
        "c1_final": C1_END,
        "c2_initial": C2_INI,
        "c2_final": C2_END,
        "search_ranges": {
            "num_leaves": list(NUM_LEAVES_RANGE),
            "min_child_samples": list(MIN_CHILD_SAMPLES_RANGE),
            "learning_rate": list(LEARNING_RATE_RANGE),
        },
        "fixed_n_estimators": N_ESTIMATORS_FIXED,
        "objective": "mean GroupKFold validation MSE in original kN units using a squared-error LightGBM model",
    },
    "quantile_levels": list(QUANTILES),
    "quantile_crossing": "Per-row quantile rearrangement (sort three predictions) applied before interval metrics and plots; raw crossing rate reported.",
    "ood_policy": "OOD is not used in training, PSO, preprocessing fit, or model selection.",
    "sample_policy": "No target cutoff and no deletion based on prediction errors.",
    "lightgbm_version": lgb.__version__,
}
with open(OUTPUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

bundle = {
    "quantile_models": quantile_models,
    "x_scaler": x_scaler,
    "y_scaler": y_scaler,
    "best_params": best_params,
    "features": FEATURES,
    "target": TARGET,
    "quantiles": QUANTILES,
    "seed": SEED,
    "source_overlap_count_dev_ood": len(source_overlap),
}
joblib.dump(bundle, OUTPUT_DIR / "lgbm_quantile_uq_model_bundle.joblib")

