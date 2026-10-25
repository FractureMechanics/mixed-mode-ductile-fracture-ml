# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FRACTURE_DATA_DIR", str(SCRIPT_DIR)))
OUTPUT_DIR = Path(os.environ.get("FRACTURE_OUTPUT_DIR", str(SCRIPT_DIR / "results_dnn_sourcewise")))
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

PSO_PARTICLES = int(os.environ.get("DNN_PSO_PARTICLES", "10"))
PSO_ITERATIONS = int(os.environ.get("DNN_PSO_ITERATIONS", "600"))
K_MAX = 600
W_INI, W_END, D1, D2 = 0.95, 0.40, 0.20, 7.0
C1_INI, C1_END = 2.5, 0.5
C2_INI, C2_END = 0.5, 2.5

N1_RANGE = (10, 100)
DROPOUT_RANGE = (0.10, 0.50)
LEARNING_RATE_RANGE = (0.001, 0.01)  

CV_EPOCHS = int(os.environ.get("DNN_CV_EPOCHS", "50"))
FINAL_EPOCHS = int(os.environ.get("DNN_FINAL_EPOCHS", "500"))
BATCH_SIZE = 32
MC_DROPOUT_PASSES = 50

SHOW_PLOTS = os.environ.get("DNN_SHOW_PLOTS", "1") == "1"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE = "#00BCD4", "#1565C0"
COLOR_TEST_FACE, COLOR_TEST_EDGE = "#2ECC71", "#0B6623"
COLOR_OOD_FACE, COLOR_OOD_EDGE = "#D81B60", "black"
COLOR_BAND = "#90A4AE"

def normalize_source_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = text.split("#", 1)[0]
    if "?" in text:
        text = text.split("?", 1)[0]
    text = text.strip().rstrip("/.")
    return text


def load_and_validate(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put {path.name} beside this script, "
            "or set FRACTURE_DATA_DIR to the folder containing both Excel files."
        )

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    required = FEATURES + [TARGET, GROUP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns {missing}. Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["_source_original"] = df[GROUP_COL].astype("string")
    df["_source_norm"] = df[GROUP_COL].map(normalize_source_id)
    if df["_source_norm"].eq("").any():
        bad_rows = df.index[df["_source_norm"].eq("")].tolist()[:10]
        raise ValueError(f"{dataset_name} has missing/blank Source_ID values at rows {bad_rows}.")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    vals = df[FEATURES + [TARGET]].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError(f"{dataset_name} contains NaN or infinite numeric values.")
    if (df[TARGET] < 0).any():
        raise ValueError(f"{dataset_name} contains negative target values; verify units/data. ")
    return df.reset_index(drop=True)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "MAPE (%)": metric_mape(y_true, y_pred),
        "RMSE (kN)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE (kN)": float(mean_absolute_error(y_true, y_pred)),
    }


def build_dnn(n1: int, dropout_rate: float, learning_rate: float, n_features: int) -> tf.keras.Model:
    """DNN architecture used by this script; three hyperparameters are tuned."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(int(n1), activation="tanh"),
            tf.keras.layers.Dropout(float(dropout_rate)),
            tf.keras.layers.Dense(32, activation="tanh"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="mse",
    )
    return model


def decode_position(position: np.ndarray) -> Dict[str, Any]:
    """Map normalized PSO position [0,1]^3 into the manuscript search ranges."""
    position = np.clip(np.asarray(position, dtype=float), 0.0, 1.0)
    n1 = int(round(N1_RANGE[0] + position[0] * (N1_RANGE[1] - N1_RANGE[0])))
    dropout = float(DROPOUT_RANGE[0] + position[1] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0]))
    lr_min, lr_max = np.log(LEARNING_RATE_RANGE[0]), np.log(LEARNING_RATE_RANGE[1])
    learning_rate = float(np.exp(lr_min + position[2] * (lr_max - lr_min)))
    return {
        "n1": int(np.clip(n1, *N1_RANGE)),
        "dropout_rate": float(np.clip(dropout, *DROPOUT_RANGE)),
        "learning_rate": float(np.clip(learning_rate, *LEARNING_RATE_RANGE)),
    }


def inertia_weight(k: int) -> float:
    """Exponential decreasing schedule consistent with the paper's Eq. (9)."""
    return float((W_INI - W_END - D1) * np.exp(K_MAX / (K_MAX + D2 * k)))


def pso_coefficients(k: int) -> Tuple[float, float]:
    c1 = C1_INI - (C1_INI - C1_END) * (k / K_MAX)
    c2 = C2_INI + (C2_END - C2_INI) * (k / K_MAX)
    return float(c1), float(c2)


def predict_raw(model: tf.keras.Model, x_scaler: PowerTransformer,
                y_scaler: PowerTransformer, X_raw: pd.DataFrame) -> np.ndarray:
    X_transformed = x_scaler.transform(X_raw[FEATURES].astype(float))
    y_pred_transformed = model.predict(X_transformed, verbose=0).reshape(-1, 1)
    return y_scaler.inverse_transform(y_pred_transformed).ravel()


def row_ids_from_df(df: pd.DataFrame) -> np.ndarray:
    for col in ROW_ID_CANDIDATES:
        if col in df.columns:
            return df[col].astype(str).to_numpy()
    return df.index.astype(str).to_numpy()

print("Loading data...")
df_dev = load_and_validate(CLEAN_FILE_PATH, "Development dataset")
df_ood = load_and_validate(UNSEEN_FILE_PATH, "OOD dataset")

print(f"Development: {len(df_dev)} samples / {df_dev['_source_norm'].nunique()} source groups")
print(f"OOD:         {len(df_ood)} samples / {df_ood['_source_norm'].nunique()} source groups")
source_overlap = set(df_dev["_source_norm"]) & set(df_ood["_source_norm"])
if source_overlap:
    raise ValueError(
        "Normalized Source_ID overlap detected between development and OOD datasets: "
        f"{sorted(source_overlap)}. Check source provenance before proceeding."
    )
print("No normalized Source_ID overlap between development and OOD.")

if df_dev["_source_norm"].nunique() < CV_FOLDS_REQUESTED + 1:
    raise ValueError("Too few unique development sources for outer split plus 5-fold GroupKFold.")

X_all = df_dev[FEATURES].copy()
y_all = df_dev[TARGET].astype(float).copy()
groups_all = df_dev["_source_norm"].copy()
outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_GROUP_FRACTION,
    random_state=SEED,
)
train_idx, test_idx = next(outer_split.split(X_all, y_all, groups=groups_all))
train_df = df_dev.iloc[train_idx].copy().reset_index(drop=True)
test_df = df_dev.iloc[test_idx].copy().reset_index(drop=True)

train_sources = set(train_df["_source_norm"])
test_sources = set(test_df["_source_norm"])
if not train_sources.isdisjoint(test_sources):
    raise RuntimeError("Source-level leakage detected between outer training and test subsets.")


X_train_raw = train_df[FEATURES].copy()
y_train_raw = train_df[TARGET].astype(float).copy()
groups_train = train_df["_source_norm"].copy()
n_splits = min(CV_FOLDS_REQUESTED, groups_train.nunique())
if n_splits < 2:
    raise ValueError("Not enough source groups in the outer training set for GroupKFold.")
cv = GroupKFold(n_splits=n_splits)
cv_splits = list(cv.split(X_train_raw, y_train_raw, groups=groups_train))

cv_fold_for_train_row = np.full(len(train_df), -1, dtype=int)
fold_rows: List[Dict[str, Any]] = []
for fold_num, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
    fit_sources = set(groups_train.iloc[fit_idx])
    val_sources = set(groups_train.iloc[val_idx])
    overlap = fit_sources & val_sources
    if overlap:
        raise RuntimeError(f"Group overlap in fold {fold_num}: {sorted(overlap)}")
    cv_fold_for_train_row[val_idx] = fold_num
    fold_rows.append({
        "Fold": fold_num,
        "Training_samples": int(len(fit_idx)),
        "Validation_samples": int(len(val_idx)),
        "Training_source_groups": int(len(fit_sources)),
        "Validation_source_groups": int(len(val_sources)),
        "Source_overlap_count": int(len(overlap)),
        "Validation_Source_IDs": "; ".join(sorted(val_sources)),
    })
if np.any(cv_fold_for_train_row < 1):
    raise RuntimeError("Not all outer training samples received a validation fold.")

objective_cache: Dict[Tuple[int, float, float], float] = {}
objective_evaluations = 0


def evaluate_candidate(params: Dict[str, Any]) -> float:
    global objective_evaluations
    cache_key = (
        int(params["n1"]),
        round(float(params["dropout_rate"]), 6),
        round(float(params["learning_rate"]), 9),
    )
    if cache_key in objective_cache:
        return objective_cache[cache_key]

    fold_r2_scores: List[float] = []
    for fold_num, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
        X_fit_raw = X_train_raw.iloc[fit_idx]
        X_val_raw = X_train_raw.iloc[val_idx]
        y_fit_raw = y_train_raw.iloc[fit_idx].to_numpy(dtype=float).reshape(-1, 1)
        y_val_raw = y_train_raw.iloc[val_idx].to_numpy(dtype=float)

        x_transformer = PowerTransformer(method="yeo-johnson")
        y_transformer = PowerTransformer(method="yeo-johnson")
        X_fit = x_transformer.fit_transform(X_fit_raw)
        X_val = x_transformer.transform(X_val_raw)
        y_fit = y_transformer.fit_transform(y_fit_raw).ravel()

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + fold_num)
        model = build_dnn(
            n1=params["n1"],
            dropout_rate=params["dropout_rate"],
            learning_rate=params["learning_rate"],
            n_features=len(FEATURES),
        )
        model.fit(
            X_fit,
            y_fit,
            epochs=CV_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=True,
        )
        pred_transformed = model.predict(X_val, verbose=0).reshape(-1, 1)
        pred_raw = y_transformer.inverse_transform(pred_transformed).ravel()
        score = r2_score(y_val_raw, pred_raw)
        if not np.isfinite(score):
            score = -1.0e6
        fold_r2_scores.append(float(score))
        del model
        tf.keras.backend.clear_session()

    objective_evaluations += 1
    mean_r2 = float(np.mean(fold_r2_scores))
    fitness = -mean_r2 
    if not np.isfinite(fitness):
        fitness = 1.0e30
    objective_cache[cache_key] = fitness
    return fitness


def run_pso() -> Tuple[Dict[str, Any], pd.DataFrame]:
    if PSO_PARTICLES < 2 or PSO_ITERATIONS < 1:
        raise ValueError("PSO_PARTICLES must be >=2 and PSO_ITERATIONS must be >=1.")

    rng = np.random.default_rng(SEED)
    dimensions = 3
    positions = rng.uniform(0.0, 1.0, size=(PSO_PARTICLES, dimensions))
    velocities = rng.uniform(-0.1, 0.1, size=(PSO_PARTICLES, dimensions))

    fitness = np.array([evaluate_candidate(decode_position(p)) for p in positions], dtype=float)
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()
    gbest_index = int(np.argmin(pbest_fitness))
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_fitness = float(pbest_fitness[gbest_index])

    history = [{
        "Iteration": 0,
        "Best_grouped_CV_R2": -gbest_fitness,
        "Iteration_best_R2": -float(np.min(fitness)),
        "Mean_particle_R2": -float(np.mean(fitness)),
        "Inertia_weight": inertia_weight(0),
        "c1": pso_coefficients(0)[0],
        "c2": pso_coefficients(0)[1],
        **decode_position(gbest_position),
    }]

    for iteration in range(1, PSO_ITERATIONS + 1):
        k = min(iteration, K_MAX)
        w = inertia_weight(k)
        c1, c2 = pso_coefficients(k)
        r1 = rng.random(size=(PSO_PARTICLES, dimensions))
        r2 = rng.random(size=(PSO_PARTICLES, dimensions))
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
        best_particle_idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[best_particle_idx] < gbest_fitness:
            gbest_fitness = float(pbest_fitness[best_particle_idx])
            gbest_position = pbest_positions[best_particle_idx].copy()

        history.append({
            "Iteration": iteration,
            "Best_grouped_CV_R2": -gbest_fitness,
            "Iteration_best_R2": -float(np.min(fitness)),
            "Mean_particle_R2": -float(np.mean(fitness)),
            "Inertia_weight": w,
            "c1": c1,
            "c2": c2,
            **decode_position(gbest_position),
        })
        if iteration == 1 or iteration % 10 == 0 or iteration == PSO_ITERATIONS:
           

    return decode_position(gbest_position), pd.DataFrame(history)



)
best_params, pso_history_df = run_pso()

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
X_scaler_final = PowerTransformer(method="yeo-johnson")
y_scaler_final = PowerTransformer(method="yeo-johnson")
X_train_scaled = X_scaler_final.fit_transform(X_train_raw)
y_train_scaled = y_scaler_final.fit_transform(y_train_raw.to_numpy(dtype=float).reshape(-1, 1)).ravel()

final_model = build_dnn(
    n1=best_params["n1"],
    dropout_rate=best_params["dropout_rate"],
    learning_rate=best_params["learning_rate"],
    n_features=len(FEATURES),
)
final_model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=FINAL_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    shuffle=True,
)

sets = {
    "Train": train_df,
    "Source-wise Test": test_df,
    "OOD": df_ood,
}
metric_rows: List[Dict[str, Any]] = []
prediction_frames: List[pd.DataFrame] = []
prediction_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
for set_name, subset in sets.items():
    y_true = subset[TARGET].to_numpy(dtype=float)
    y_pred = predict_raw(final_model, X_scaler_final, y_scaler_final, subset[FEATURES])
    prediction_cache[set_name] = (y_true, y_pred)
    metrics = calculate_metrics(y_true, y_pred)
    metric_rows.append({
        "Dataset": set_name,
        "N": int(len(subset)),
        "Source_groups": int(subset["_source_norm"].nunique()),
        **metrics,
    })
    prediction_frames.append(pd.DataFrame({
        "Dataset": set_name,
        "Sample_ID": row_ids_from_df(subset),
        "Source_ID": subset["_source_original"].astype(str).to_numpy(),
        "Actual fracture load (kN)": y_true,
        "Predicted fracture load (kN)": y_pred,
        "Residual (Actual - Predicted) (kN)": y_true - y_pred,
        "Absolute error (kN)": np.abs(y_true - y_pred),
    }))

metrics_df = pd.DataFrame(metric_rows)
predictions_df = pd.concat(prediction_frames, ignore_index=True)
print("\nFinal DNN performance (all samples included):")
print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

print("\nComputing source-wise out-of-fold predictions for residual analysis...")
oof_actual = np.full(len(train_df), np.nan, dtype=float)
oof_predicted = np.full(len(train_df), np.nan, dtype=float)
for fold_num, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
    X_fit_raw = X_train_raw.iloc[fit_idx]
    X_val_raw = X_train_raw.iloc[val_idx]
    y_fit_raw = y_train_raw.iloc[fit_idx].to_numpy(dtype=float).reshape(-1, 1)
    x_transformer = PowerTransformer(method="yeo-johnson")
    y_transformer = PowerTransformer(method="yeo-johnson")
    X_fit = x_transformer.fit_transform(X_fit_raw)
    X_val = x_transformer.transform(X_val_raw)
    y_fit = y_transformer.fit_transform(y_fit_raw).ravel()

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + 1000 + fold_num)
    fold_model = build_dnn(
        n1=best_params["n1"],
        dropout_rate=best_params["dropout_rate"],
        learning_rate=best_params["learning_rate"],
        n_features=len(FEATURES),
    )
    fold_model.fit(X_fit, y_fit, epochs=FINAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
    fold_pred_transformed = fold_model.predict(X_val, verbose=0).reshape(-1, 1)
    fold_pred_raw = y_transformer.inverse_transform(fold_pred_transformed).ravel()
    oof_actual[val_idx] = y_train_raw.iloc[val_idx].to_numpy(dtype=float)
    oof_predicted[val_idx] = fold_pred_raw
    del fold_model
    tf.keras.backend.clear_session()

if np.isnan(oof_predicted).any():
    raise RuntimeError("Some OOF predictions were not generated.")
oof_residuals = oof_actual - oof_predicted
resid_q_low, resid_q_high = np.quantile(oof_residuals, [0.025, 0.975])
oof_residual_df = pd.DataFrame({
    "Sample_ID": row_ids_from_df(train_df),
    "Source_ID": train_df["_source_original"].astype(str).to_numpy(),
    "Fold": cv_fold_for_train_row,
    "Actual_load_kN": oof_actual,
    "OOF_predicted_load_kN": oof_predicted,
    "OOF_residual_actual_minus_predicted_kN": oof_residuals,
})

def plot_scatter_panels(zoom: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    specs = [
        ("Train", COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE),
        ("Source-wise Test", COLOR_TEST_FACE, COLOR_TEST_EDGE),
        ("OOD", COLOR_OOD_FACE, COLOR_OOD_EDGE),
    ]
    if zoom:
        axis_max = 60.0
        title = "DNN actual vs. predicted (visual zoom ≤60 kN; metrics use all samples)"
        filename = OUTPUT_DIR / "DNN_Sourcewise_Scatter_Zoom_0_60kN.png"
    else:
        vals = [
            np.concatenate(prediction_cache[name])
            for name, _, _ in specs
        ]
        axis_max = max(1.0, float(np.nanmax(np.concatenate(vals))) * 1.03)
        title = "DNN actual vs. predicted fracture loads (all samples)"
        filename = OUTPUT_DIR / "DNN_Sourcewise_Scatter_AllData.png"

    line_x = np.linspace(0.0, axis_max, 250)
    band_low = line_x - resid_q_high
    band_high = line_x - resid_q_low

    for ax, (set_name, face_color, edge_color) in zip(axes, specs):
        y_true, y_pred = prediction_cache[set_name]
        if zoom:
            visible = (y_true >= 0) & (y_true <= 60) & (y_pred >= 0) & (y_pred <= 60)
        else:
            visible = np.ones(len(y_true), dtype=bool)

        ax.fill_between(
            line_x, band_low, band_high, color=COLOR_BAND, alpha=0.25,
            linewidth=0, label="Approx. 95% global OOF residual band", zorder=1,
        )
        ax.scatter(
            y_true[visible], y_pred[visible],
            facecolors=face_color, edgecolors=edge_color,
            s=65, alpha=0.85, linewidths=1.0, label="Observations", zorder=3,
        )
        ax.plot([0, axis_max], [0, axis_max], "k--", lw=1.8, label="Ideal line", zorder=2)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(set_name, fontweight="bold")
        ax.set_xlabel("Actual fracture load (kN)")
        ax.set_ylabel("Predicted fracture load (kN)")
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(loc="lower right", fontsize=8, frameon=True, edgecolor="black")
        metric = next(row for row in metric_rows if row["Dataset"] == set_name)
        ax.text(
            0.04, 0.96,
            f"$R^2={metric['R2']:.3f}$\nMAPE={metric['MAPE (%)']:.2f}%\nN={metric['N']}",
            transform=ax.transAxes, va="top", ha="left", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, boxstyle="round,pad=0.3"),
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


plot_scatter_panels(zoom=False)
plot_scatter_panels(zoom=True)

print(f"\nRunning {MC_DROPOUT_PASSES}-pass MC dropout on OOD data...")
X_ood_scaled = X_scaler_final.transform(df_ood[FEATURES].astype(float))
mc_predictions_scaled = []
for pass_num in range(MC_DROPOUT_PASSES):
    seed_for_pass = SEED + 5000 + pass_num
    tf.keras.utils.set_random_seed(seed_for_pass)
    pred_scaled = final_model(X_ood_scaled, training=True).numpy().reshape(-1, 1)
    pred_raw = y_scaler_final.inverse_transform(pred_scaled).ravel()
    mc_predictions_scaled.append(pred_raw)
mc_predictions = np.vstack(mc_predictions_scaled)
mc_mean = np.mean(mc_predictions, axis=0)
mc_std = np.std(mc_predictions, axis=0, ddof=1)
mc_metrics = calculate_metrics(df_ood[TARGET].to_numpy(dtype=float), mc_mean)

mc_dropout_df = pd.DataFrame({
    "Sample_ID": row_ids_from_df(df_ood),
    "Source_ID": df_ood["_source_original"].astype(str).to_numpy(),
    "Actual fracture load (kN)": df_ood[TARGET].to_numpy(dtype=float),
    "MC-dropout mean prediction (kN)": mc_mean,
    "MC-dropout standard deviation (kN)": mc_std,
    "Mean minus 1 SD (kN)": mc_mean - mc_std,
    "Mean plus 1 SD (kN)": mc_mean + mc_std,
})
mc_metrics_row = {
    "Dataset": "OOD (MC-dropout predictive mean)",
    "N": int(len(df_ood)),
    "Source_groups": int(df_ood["_source_norm"].nunique()),
    **mc_metrics,
}
metrics_with_mc_df = pd.concat([metrics_df, pd.DataFrame([mc_metrics_row])], ignore_index=True)

fig, ax = plt.subplots(figsize=(9, 6.5))
order = np.argsort(df_ood[TARGET].to_numpy(dtype=float))
y_true_ood = df_ood[TARGET].to_numpy(dtype=float)[order]
y_mc = mc_mean[order]
y_sd = mc_std[order]
ax.errorbar(
    y_true_ood, y_mc, yerr=y_sd, fmt="o", markersize=4.5,
    capsize=2, alpha=0.75, color="#D81B60", ecolor="#90A4AE",
    label="MC-dropout mean ± 1 SD",
)
all_max = max(1.0, float(max(np.max(y_true_ood), np.max(y_mc + y_sd))) * 1.05)
ax.plot([0, all_max], [0, all_max], "k--", lw=1.8, label="Ideal line")
ax.set_xlim(0, all_max)
ax.set_ylim(0, all_max)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Actual fracture load (kN)")
ax.set_ylabel("MC-dropout mean prediction (kN)")
ax.set_title("DNN MC-dropout on independent OOD data", fontweight="bold")
ax.text(
    0.04, 0.96,
    f"$R^2={mc_metrics['R2']:.3f}$\nMAPE={mc_metrics['MAPE (%)']:.2f}%\nN={len(df_ood)}",
    transform=ax.transAxes, va="top", ha="left", fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, boxstyle="round,pad=0.3"),
)
ax.legend(frameon=True)
ax.tick_params(direction="in", top=True, right=True)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "DNN_MC_Dropout_OOD.png", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(
    pso_history_df["Iteration"], pso_history_df["Best_grouped_CV_R2"],
    color="#1565C0", linewidth=2.2,
)
ax.set_title("DNN PSO convergence (source-wise GroupKFold)", fontweight="bold")
ax.set_xlabel("PSO iteration")
ax.set_ylabel("Best grouped-CV $R^2$ (original load scale)")
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "DNN_PSO_Convergence.png", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

iteration_grid = np.arange(0, K_MAX + 1)
w_grid = np.array([inertia_weight(int(k)) for k in iteration_grid])
c1_grid = np.array([pso_coefficients(int(k))[0] for k in iteration_grid])
c2_grid = np.array([pso_coefficients(int(k))[1] for k in iteration_grid])
fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_w.plot(iteration_grid, w_grid, linewidth=2.4, color="#5E35B1")
ax_w.set_xlabel("Iteration")
ax_w.set_ylabel("Inertia weight")
ax_w.set_title("Inertia-weight schedule")
ax_c.plot(iteration_grid, c1_grid, linewidth=2.4, color="#3949AB", label="$c_1$")
ax_c.plot(iteration_grid, c2_grid, linewidth=2.4, color="#FF4081", label="$c_2$")
ax_c.set_xlabel("Iteration")
ax_c.set_ylabel("Acceleration coefficient")
ax_c.set_title("Acceleration-coefficient schedules")
ax_c.legend(frameon=False)
for ax in (ax_w, ax_c):
    ax.tick_params(direction="in")
    ax.grid(False)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "DNN_PSO_Schedules.png", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

partition = pd.DataFrame({
    "Development_row_index": np.arange(len(df_dev)),
    "Sample_ID": row_ids_from_df(df_dev),
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
    "Sample_ID": row_ids_from_df(df_ood),
    "Source_ID": df_ood["_source_original"].astype(str).to_numpy(),
    "Source_ID_normalized": df_ood["_source_norm"].to_numpy(),
    "Split": "OOD",
})
fold_summary_df = pd.DataFrame(fold_rows)
source_count_df = (
    pd.concat([
        partition[["Source_ID_normalized", "Split"]].rename(columns={"Source_ID_normalized": "Source_ID"}),
        ood_partition[["Source_ID_normalized", "Split"]].rename(columns={"Source_ID_normalized": "Source_ID"}),
    ], ignore_index=True)
    .groupby(["Split", "Source_ID"], dropna=False)
    .size().reset_index(name="N_samples")
)

final_model_path = OUTPUT_DIR / "DNN_Sourcewise_Model.keras"
final_model.save(final_model_path)
joblib.dump(
    {"x_scaler": X_scaler_final, "y_scaler": y_scaler_final, "features": FEATURES,
     "target": TARGET, "best_params": best_params, "seed": SEED,
     "sourcewise_train_source_ids": sorted(train_sources),
     "sourcewise_test_source_ids": sorted(test_sources),
     "ood_source_ids": sorted(set(df_ood["_source_norm"]))},
    OUTPUT_DIR / "DNN_Sourcewise_Preprocessing_and_Config.joblib",
)

config = {
    "seed": SEED,
    "development_file": CLEAN_FILE_PATH.name,
    "ood_file": UNSEEN_FILE_PATH.name,
    "features": FEATURES,
    "target": TARGET,
    "grouping_variable": GROUP_COL,
    "outer_split": "GroupShuffleSplit",
    "outer_test_group_fraction": TEST_GROUP_FRACTION,
    "cv": f"GroupKFold(n_splits={n_splits})",
    "source_overlap_development_ood": 0,
    "pso_population": PSO_PARTICLES,
    "pso_iterations": PSO_ITERATIONS,
    "pso_schedule": {
        "w_ini": W_INI, "w_end": W_END, "d1": D1, "d2": D2, "k_max": K_MAX,
        "c1_ini": C1_INI, "c1_end": C1_END, "c2_ini": C2_INI, "c2_end": C2_END,
    },
    "cv_epochs_for_pso_objective": CV_EPOCHS,
    "final_epochs": FINAL_EPOCHS,
    "oof_residual_calibration_epochs": FINAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "mc_dropout_passes": MC_DROPOUT_PASSES,
    "search_ranges": {
        "first_hidden_layer_neurons": list(N1_RANGE),
        "dropout_rate": list(DROPOUT_RANGE),
        "learning_rate_log_uniform": list(LEARNING_RATE_RANGE),
    },
    "fixed_architecture": ["Dense(n1,tanh)", "Dropout(rate)", "Dense(32,tanh)", "Dense(32,relu)", "Dense(1,linear)"],
    "best_params": best_params,
    "metrics_include_all_samples": True,
    "ood_used_for_training_or_tuning": False,
    "notes": [
        "No fracture-load cutoff applied.",
        "No rows removed based on prediction residuals.",
        "MC-dropout standard deviation approximates epistemic/model uncertainty; it is not total predictive uncertainty.",
        "The 0-60 kN plot is a visual zoom only; all observations are included in metrics.",
    ],
}
(OUTPUT_DIR / "DNN_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


