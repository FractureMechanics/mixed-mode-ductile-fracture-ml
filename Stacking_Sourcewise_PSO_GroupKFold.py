# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get( str(SCRIPT_DIR)))
OUTPUT_DIR = Path(os.environ.get( str(SCRIPT_DIR / "results_stacking_sourcewise")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PSO_CONVERGENCE_PNG = OUTPUT_DIR / "Stacking_LightGBM_PSO_Convergence.png"
PSO_SCHEDULE_PNG = OUTPUT_DIR / "Stacking_PSO_Schedules.png"
SCATTER_FULL_PNG = OUTPUT_DIR / "Stacking_Sourcewise_Scatter_AllData.png"
SCATTER_ZOOM_PNG = OUTPUT_DIR / "Stacking_Sourcewise_Scatter_Zoom_0_60kN.png"
MODEL_BUNDLE_PATH = OUTPUT_DIR / "Stacking_Sourcewise_ModelBundle.joblib"
DNN_MODEL_PATH = OUTPUT_DIR / "Stacking_Sourcewise_DNN.keras"
RUN_CONFIG_PATH = OUTPUT_DIR / "Stacking_Sourcewise_RunConfig.json"

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

PSO_PARTICLES = int(os.environ.get("STACK_PSO_PARTICLES", "10"))
PSO_ITERATIONS = int(os.environ.get("STACK_PSO_ITERATIONS", "600"))
K_MAX = 600
W_INI, W_END, D1, D2 = 0.95, 0.40, 0.20, 7.0
C1_INI, C1_END = 2.5, 0.5
C2_INI, C2_END = 0.5, 2.5

NUM_LEAVES_RANGE = (10, 100)
MIN_CHILD_SAMPLES_RANGE = (5, 50)
LEARNING_RATE_RANGE = (0.01, 0.30)  
N_ESTIMATORS_FIXED = 100

DNN_EPOCHS = int(os.environ.get("STACK_DNN_EPOCHS", "600"))
DNN_BATCH_SIZE = int(os.environ.get("STACK_DNN_BATCH_SIZE", "32"))
DNN_N1 = 46
DNN_DROPOUT = 0.31
DNN_LEARNING_RATE = 0.0039
DNN_SECOND_LAYER = 122
DNN_THIRD_LAYER = 32

XGB_N_ESTIMATORS = 555
XGB_LEARNING_RATE = 0.016
XGB_MAX_DEPTH = 14
XGB_N_JOBS = int(os.environ.get("STACK_XGB_N_JOBS", "-1"))
LGBM_N_JOBS = int(os.environ.get("STACK_LGBM_N_JOBS", "-1"))
SHOW_PLOTS = os.environ.get("STACK_SHOW_PLOTS", "1") == "1"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE = "#00BCD4", "#1565C0"
COLOR_TEST_FACE, COLOR_TEST_EDGE = "#2ECC71", "#0B6623"
COLOR_OOD_FACE, COLOR_OOD_EDGE = "#D81B60", "black"
COLOR_BAND = "#90A4AE"

def normalize_source_id(value: Any) -> str:
    """Normalize DOI / source identifiers to catch URL-vs-DOI formatting differences."""
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    if text.startswith(("http://", "https://")):
        parsed = urlsplit(text)
        text = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    return text.strip().rstrip("/.")


def load_and_validate(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put {path.name} next to this script or set FRACTURE_DATA_DIR."
        )
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    required = FEATURES + [TARGET, GROUP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns {missing}. Available columns: {df.columns.tolist()}")

    df = df.copy()
    id_col = next((col for col in ROW_ID_CANDIDATES if col in df.columns), None)
    df["_sample_id"] = df[id_col].astype(str) if id_col else (np.arange(len(df)) + 1).astype(str)
    df["_source_original"] = df[GROUP_COL].astype("string")
    df["_source_norm"] = df[GROUP_COL].map(normalize_source_id)
    if df["_source_norm"].isna().any() or df["_source_norm"].eq("").any():
        bad = df.index[df["_source_norm"].isna() | df["_source_norm"].eq("")].tolist()[:10]
        raise ValueError(f"{dataset_name} has missing/blank Source_ID at row indices {bad}.")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    values = df[FEATURES + [TARGET]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset_name} contains NaN or infinite feature/target values.")
    if (df[TARGET] < 0).any():
        raise ValueError(f"{dataset_name} contains negative fracture-load values; verify source data/units.")
    return df.reset_index(drop=True)


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.abs(y_true) > np.finfo(float).eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "N": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "MAPE (%)": mape_percent(y_true, y_pred),
        "RMSE (kN)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE (kN)": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_nonzero_N": int(np.count_nonzero(np.abs(y_true) > np.finfo(float).eps)),
    }


def build_dnn(n_features: int) -> tf.keras.Model:
    """DNN architecture/settings retained from the supplied stacking code."""
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(
                DNN_N1, activation="tanh",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 1),
            ),
            tf.keras.layers.Dropout(DNN_DROPOUT, seed=SEED + 2),
            tf.keras.layers.Dense(
                DNN_SECOND_LAYER, activation="tanh",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 3),
            ),
            tf.keras.layers.Dense(
                DNN_THIRD_LAYER, activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 4),
            ),
            tf.keras.layers.Dense(
                1, activation="linear",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 5),
            ),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(DNN_LEARNING_RATE), loss="mse")
    return model


def build_lgbm(params: Dict[str, Any]) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS_FIXED,
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        learning_rate=float(params["learning_rate"]),
        random_state=SEED,
        n_jobs=LGBM_N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def build_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        random_state=SEED,
        n_jobs=XGB_N_JOBS,
        verbosity=0,
    )


def decode_position(position: np.ndarray) -> Dict[str, Any]:
    pos = np.clip(np.asarray(position, dtype=float), 0.0, 1.0)
    leaves = int(round(NUM_LEAVES_RANGE[0] + pos[0] * (NUM_LEAVES_RANGE[1] - NUM_LEAVES_RANGE[0])))
    min_child = int(round(MIN_CHILD_SAMPLES_RANGE[0] + pos[1] * (MIN_CHILD_SAMPLES_RANGE[1] - MIN_CHILD_SAMPLES_RANGE[0])))
    log_min, log_max = np.log(LEARNING_RATE_RANGE[0]), np.log(LEARNING_RATE_RANGE[1])
    lr = float(np.exp(log_min + pos[2] * (log_max - log_min)))
    return {
        "num_leaves": int(np.clip(leaves, *NUM_LEAVES_RANGE)),
        "min_child_samples": int(np.clip(min_child, *MIN_CHILD_SAMPLES_RANGE)),
        "learning_rate": float(np.clip(lr, *LEARNING_RATE_RANGE)),
    }


def inertia_weight(k: int) -> float:
    return float((W_INI - W_END - D1) * np.exp(K_MAX / (K_MAX + D2 * k)))


def pso_coefficients(k: int) -> Tuple[float, float]:
    c1 = C1_INI - (C1_INI - C1_END) * (k / K_MAX)
    c2 = C2_INI + (C2_END - C2_INI) * (k / K_MAX)
    return float(c1), float(c2)

df_dev = load_and_validate(CLEAN_FILE_PATH)
df_ood = load_and_validate(OOD_FILE_PATH)

source_overlap = sorted(set(df_dev["_source_norm"]) & set(df_ood["_source_norm"]))
if source_overlap:
    raise ValueError(
        "Normalized publication Source_ID overlap detected between development and OOD. "
        "Resolve source provenance before claiming source-wise independence: " + "; ".join(source_overlap)
    )
if df_dev["_source_norm"].nunique() < CV_FOLDS_REQUESTED + 1:
    raise ValueError("Not enough unique source groups for the requested outer split and GroupKFold.")

outer_split = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRACTION, random_state=SEED)
train_idx, test_idx = next(
    outer_split.split(df_dev[FEATURES], df_dev[TARGET], groups=df_dev["_source_norm"])
)
train_df = df_dev.iloc[train_idx].copy().reset_index(drop=True)
test_df = df_dev.iloc[test_idx].copy().reset_index(drop=True)

train_sources = set(train_df["_source_norm"])
test_sources = set(test_df["_source_norm"])
if not train_sources.isdisjoint(test_sources):
    raise RuntimeError("Source-level overlap detected between source-wise Train and Test.")
X_train_raw = train_df[FEATURES].copy()
y_train_raw = train_df[TARGET].astype(float).to_numpy()
groups_train = train_df["_source_norm"].copy()
X_test_raw = test_df[FEATURES].copy()
y_test_raw = test_df[TARGET].astype(float).to_numpy()
X_ood_raw = df_ood[FEATURES].copy()
y_ood_raw = df_ood[TARGET].astype(float).to_numpy()

n_splits = min(CV_FOLDS_REQUESTED, groups_train.nunique())
if n_splits < 2:
    raise ValueError("Not enough source groups in outer Train for GroupKFold.")
cv = GroupKFold(n_splits=n_splits)
cv_splits = list(cv.split(X_train_raw, y_train_raw, groups=groups_train))

fold_audit_rows: List[Dict[str, Any]] = []
fold_id_for_train_row = np.zeros(len(train_df), dtype=int)
for fold_no, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
    fit_groups = set(groups_train.iloc[fit_idx])
    val_groups = set(groups_train.iloc[val_idx])
    overlap = fit_groups & val_groups
    if overlap:
        raise RuntimeError(f"Group overlap in GroupKFold fold {fold_no}: {sorted(overlap)}")
    fold_id_for_train_row[val_idx] = fold_no
    fold_audit_rows.append({
        "Fold": fold_no,
        "Fit_samples": len(fit_idx),
        "Validation_samples": len(val_idx),
        "Fit_source_groups": len(fit_groups),
        "Validation_source_groups": len(val_groups),
        "Source_overlap_count": len(overlap),
        "Validation_Source_IDs": "; ".join(sorted(val_groups)),
    })
if np.any(fold_id_for_train_row == 0):
    raise RuntimeError("Not every outer Train row was assigned to an OOF validation fold.")

objective_cache: Dict[Tuple[int, int, float], float] = {}
objective_evaluations = 0


def evaluate_lgbm_candidate(params: Dict[str, Any]) -> float:
    """Mean fold MSE on fold-specific Yeo-Johnson-transformed target values."""
    global objective_evaluations
    key = (int(params["num_leaves"]), int(params["min_child_samples"]), round(float(params["learning_rate"]), 8))
    if key in objective_cache:
        return objective_cache[key]

    fold_losses: List[float] = []
    for fold_no, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
        X_fit_raw = X_train_raw.iloc[fit_idx]
        X_val_raw = X_train_raw.iloc[val_idx]
        y_fit_raw = y_train_raw[fit_idx].reshape(-1, 1)
        y_val_raw = y_train_raw[val_idx].reshape(-1, 1)

        x_scaler = PowerTransformer(method="yeo-johnson")
        y_scaler = PowerTransformer(method="yeo-johnson")
        X_fit = x_scaler.fit_transform(X_fit_raw)
        X_val = x_scaler.transform(X_val_raw)
        y_fit = y_scaler.fit_transform(y_fit_raw).ravel()
        y_val = y_scaler.transform(y_val_raw).ravel()

        model = build_lgbm(params)
        model.fit(X_fit, y_fit)
        pred = np.asarray(model.predict(X_val), dtype=float).ravel()
        loss = float(mean_squared_error(y_val, pred))
        if not np.isfinite(loss):
            loss = 1.0e30
        fold_losses.append(loss)

    objective_evaluations += 1
    result = float(np.mean(fold_losses))
    objective_cache[key] = result
    return result


def run_pso() -> Tuple[Dict[str, Any], pd.DataFrame]:
    if PSO_PARTICLES < 2 or PSO_ITERATIONS < 1:
        raise ValueError("STACK_PSO_PARTICLES must be >=2 and STACK_PSO_ITERATIONS must be >=1.")

    rng = np.random.default_rng(SEED)
    dimensions = 3
    positions = rng.uniform(0.0, 1.0, size=(PSO_PARTICLES, dimensions))
    velocities = rng.uniform(-0.1, 0.1, size=(PSO_PARTICLES, dimensions))
    fitness = np.array([evaluate_lgbm_candidate(decode_position(p)) for p in positions], dtype=float)
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()
    best_idx = int(np.argmin(pbest_fitness))
    gbest_position = pbest_positions[best_idx].copy()
    gbest_fitness = float(pbest_fitness[best_idx])

    history: List[Dict[str, Any]] = [{
        "Iteration": 0,
        "Global_best_grouped_CV_MSE_transformed_target": gbest_fitness,
        "Iteration_best_grouped_CV_MSE_transformed_target": float(np.min(fitness)),
        "Mean_particle_grouped_CV_MSE_transformed_target": float(np.mean(fitness)),
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
        fitness = np.array([evaluate_lgbm_candidate(decode_position(p)) for p in positions], dtype=float)
        improved = fitness < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = fitness[improved]
        candidate_idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[candidate_idx] < gbest_fitness:
            gbest_fitness = float(pbest_fitness[candidate_idx])
            gbest_position = pbest_positions[candidate_idx].copy()

        history.append({
            "Iteration": iteration,
            "Global_best_grouped_CV_MSE_transformed_target": gbest_fitness,
            "Iteration_best_grouped_CV_MSE_transformed_target": float(np.min(fitness)),
            "Mean_particle_grouped_CV_MSE_transformed_target": float(np.mean(fitness)),
            "Inertia_weight": w,
            "c1": c1,
            "c2": c2,
            **decode_position(gbest_position),
        })
        if iteration == 1 or iteration % 10 == 0 or iteration == PSO_ITERATIONS:
          
    return decode_position(gbest_position), pd.DataFrame(history)
best_lgbm_params, pso_history_df = run_pso()

oof_base_predictions = np.full((len(train_df), 3), np.nan, dtype=float)

for fold_no, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
    print(f"\nGenerating OOF base predictions: fold {fold_no}/{n_splits}")
    X_fit_raw = X_train_raw.iloc[fit_idx]
    X_val_raw = X_train_raw.iloc[val_idx]
    y_fit_raw = y_train_raw[fit_idx].reshape(-1, 1)

    x_scaler_fold = PowerTransformer(method="yeo-johnson")
    y_scaler_fold = PowerTransformer(method="yeo-johnson")
    X_fit = x_scaler_fold.fit_transform(X_fit_raw)
    X_val = x_scaler_fold.transform(X_val_raw)
    y_fit = y_scaler_fold.fit_transform(y_fit_raw).ravel()

    model_lgbm_fold = build_lgbm(best_lgbm_params)
    model_lgbm_fold.fit(X_fit, y_fit)
    lgbm_pred_s = np.asarray(model_lgbm_fold.predict(X_val), dtype=float).reshape(-1, 1)
    lgbm_pred = y_scaler_fold.inverse_transform(lgbm_pred_s).ravel()

    model_xgb_fold = build_xgb()
    model_xgb_fold.fit(X_fit, y_fit)
    xgb_pred_s = np.asarray(model_xgb_fold.predict(X_val), dtype=float).reshape(-1, 1)
    xgb_pred = y_scaler_fold.inverse_transform(xgb_pred_s).ravel()

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + fold_no)
    model_dnn_fold = build_dnn(len(FEATURES))
    model_dnn_fold.fit(
        X_fit, y_fit,
        epochs=DNN_EPOCHS,
        batch_size=DNN_BATCH_SIZE,
        verbose=0,
        shuffle=True,
    )
    dnn_pred_s = np.asarray(model_dnn_fold.predict(X_val, verbose=0), dtype=float).reshape(-1, 1)
    dnn_pred = y_scaler_fold.inverse_transform(dnn_pred_s).ravel()

    oof_base_predictions[val_idx, :] = np.column_stack([dnn_pred, lgbm_pred, xgb_pred])
    del model_dnn_fold, model_lgbm_fold, model_xgb_fold
    tf.keras.backend.clear_session()

if not np.isfinite(oof_base_predictions).all():
    raise RuntimeError("OOF base predictions contain missing/non-finite values.")

meta_model = LinearRegression()
meta_model.fit(oof_base_predictions, y_train_raw)
oof_meta_train_pred = meta_model.predict(oof_base_predictions).ravel()

x_scaler_final = PowerTransformer(method="yeo-johnson")
y_scaler_final = PowerTransformer(method="yeo-johnson")
X_train_scaled = x_scaler_final.fit_transform(X_train_raw)
y_train_scaled = y_scaler_final.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
X_test_scaled = x_scaler_final.transform(X_test_raw)
X_ood_scaled = x_scaler_final.transform(X_ood_raw)

final_lgbm = build_lgbm(best_lgbm_params)
final_lgbm.fit(X_train_scaled, y_train_scaled)
final_xgb = build_xgb()
final_xgb.fit(X_train_scaled, y_train_scaled)

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED + 2026)
final_dnn = build_dnn(len(FEATURES))
final_dnn.fit(
    X_train_scaled, y_train_scaled,
    epochs=DNN_EPOCHS,
    batch_size=DNN_BATCH_SIZE,
    verbose=0,
    shuffle=True,
)


def stack_predict(X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base_matrix_scaled = np.column_stack([
        np.asarray(final_dnn.predict(X_scaled, verbose=0), dtype=float).ravel(),
        np.asarray(final_lgbm.predict(X_scaled), dtype=float).ravel(),
        np.asarray(final_xgb.predict(X_scaled), dtype=float).ravel(),
    ])
    base_matrix_raw = np.column_stack([
        y_scaler_final.inverse_transform(base_matrix_scaled[:, [j]]).ravel()
        for j in range(base_matrix_scaled.shape[1])
    ])
    stacked = meta_model.predict(base_matrix_raw).ravel()
    return stacked, base_matrix_raw


test_pred, test_base_pred = stack_predict(X_test_scaled)
ood_pred, ood_base_pred = stack_predict(X_ood_scaled)

metrics_rows: List[Dict[str, Any]] = []

def prediction_table(df: pd.DataFrame, dataset_name: str, y_true: np.ndarray,
                     y_pred: np.ndarray, base_predictions: np.ndarray | None = None) -> pd.DataFrame:
    out = pd.DataFrame({
        "Dataset": dataset_name,
        "Sample_ID": df["_sample_id"].astype(str).to_numpy(),
        "Source_ID_original": df["_source_original"].astype(str).to_numpy(),
        "Source_ID_normalized": df["_source_norm"].astype(str).to_numpy(),
        "GroupKFold_OOF_Fold": fold_id_for_train_row if dataset_name.startswith("Train") else np.nan,
        "Actual fracture load (kN)": np.asarray(y_true, dtype=float),
        "Stacking predicted fracture load (kN)": np.asarray(y_pred, dtype=float),
        "Residual actual-minus-predicted (kN)": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
        "Absolute percentage error (%)": np.where(
            np.abs(np.asarray(y_true, dtype=float)) > np.finfo(float).eps,
            np.abs((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) /
                 np.where(np.abs(np.asarray(y_true, dtype=float)) > np.finfo(float).eps, np.asarray(y_true, dtype=float), 1.0)) * 100.0,
            np.nan,
        ),
    })
    if base_predictions is not None:
        out["DNN predicted (kN)"] = base_predictions[:, 0]
        out["LightGBM predicted (kN)"] = base_predictions[:, 1]
        out["XGBoost predicted (kN)"] = base_predictions[:, 2]
    return out

train_pred_table = prediction_table(
    train_df, "Train (OOF base features; meta-model fitted on these rows)",
    y_train_raw, oof_meta_train_pred, oof_base_predictions,
)
test_pred_table = prediction_table(test_df, "Source-wise Test", y_test_raw, test_pred, test_base_pred)
ood_pred_table = prediction_table(df_ood, "Independent OOD", y_ood_raw, ood_pred, ood_base_pred)
predictions_df = pd.concat([train_pred_table, test_pred_table, ood_pred_table], ignore_index=True)

oof_residuals = y_train_raw - oof_meta_train_pred
resid_q025, resid_q975 = np.quantile(oof_residuals, [0.025, 0.975])

def draw_scatter_panel(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str,
                       face_color: str, edge_color: str, axis_limit: float | None = None,
                       include_band: bool = True) -> None:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    ax.scatter(yt, yp, fc=face_color, ec=edge_color, s=70, alpha=0.88, lw=1.2,
               zorder=5, label="Data points")
    finite_all = np.concatenate([yt[np.isfinite(yt)], yp[np.isfinite(yp)]])
    if axis_limit is None:
        max_val = float(np.nanmax(finite_all)) if finite_all.size else 1.0
        min_val = float(np.nanmin(finite_all)) if finite_all.size else 0.0
        low = min(0.0, min_val)
        high = max(1.0, max_val) * 1.04
    else:
        low, high = 0.0, float(axis_limit)
    line = np.linspace(low, high, 200)
    ax.plot(line, line, "k--", lw=2.0, label="Ideal line", zorder=4)
    if include_band:
        lower = line - resid_q975
        upper = line - resid_q025
        ax.fill_between(line, lower, upper, color=COLOR_BAND, alpha=0.22, lw=0,
                        label="Empirical 95% OOF residual interval")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Load (kN)")
    ax.set_ylabel("Predicted Load (kN)")
    ax.tick_params(direction="in", top=True, right=True)
    metric = compute_metrics(yt, yp)
    stats_text = f"$R^2 = {metric['R2']:.3f}$\n$MAPE = {metric['MAPE (%)']:.2f}\\%$"
    ax.text(0.04, 0.95, stats_text, transform=ax.transAxes, va="top", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
    ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=False, fontsize=8)


fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
draw_scatter_panel(axes[0], y_train_raw, oof_meta_train_pred, "Train: OOF Base Features",
                   COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE, axis_limit=None)
draw_scatter_panel(axes[1], y_test_raw, test_pred, "Source-wise Test",
                   COLOR_TEST_FACE, COLOR_TEST_EDGE, axis_limit=None)
draw_scatter_panel(axes[2], y_ood_raw, ood_pred, "Independent OOD",
                   COLOR_OOD_FACE, COLOR_OOD_EDGE, axis_limit=None)
fig.tight_layout(pad=2.0)
fig.savefig(SCATTER_FULL_PNG, dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

fig_zoom, axes_zoom = plt.subplots(1, 3, figsize=(19, 6.5))
draw_scatter_panel(axes_zoom[0], y_train_raw, oof_meta_train_pred, "Train: OOF Base Features (0–60 kN)",
                   COLOR_TRAIN_FACE, COLOR_TRAIN_EDGE, axis_limit=60.0)
draw_scatter_panel(axes_zoom[1], y_test_raw, test_pred, "Source-wise Test (0–60 kN)",
                   COLOR_TEST_FACE, COLOR_TEST_EDGE, axis_limit=60.0)
draw_scatter_panel(axes_zoom[2], y_ood_raw, ood_pred, "Independent OOD (0–60 kN)",
                   COLOR_OOD_FACE, COLOR_OOD_EDGE, axis_limit=60.0)
fig_zoom.tight_layout(pad=2.0)
fig_zoom.savefig(SCATTER_ZOOM_PNG, dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig_zoom)

fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.plot(pso_history_df["Iteration"], pso_history_df["Global_best_grouped_CV_MSE_transformed_target"],
        color="#1565C0", lw=2.3, marker="o", markersize=3.2, label="Global-best grouped-CV MSE")
ax.scatter(pso_history_df["Iteration"], pso_history_df["Iteration_best_grouped_CV_MSE_transformed_target"],
           color="#D81B60", alpha=0.55, s=20, label="Iteration-best MSE")
ax.set_title("LightGBM PSO Convergence for Stacking", fontweight="bold")
ax.set_xlabel("PSO iteration")
ax.set_ylabel("Mean fold MSE (transformed target space)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, edgecolor="black", fancybox=False)
fig.tight_layout()
fig.savefig(PSO_CONVERGENCE_PNG, dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

iter_axis = np.arange(0, K_MAX + 1)
w_vals = np.array([inertia_weight(int(k)) for k in iter_axis])
c1_vals = np.array([pso_coefficients(int(k))[0] for k in iter_axis])
c2_vals = np.array([pso_coefficients(int(k))[1] for k in iter_axis])
fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
ax1.plot(iter_axis, w_vals, lw=2.0, label="Inertia weight w")
ax1.plot(iter_axis, c1_vals, lw=2.0, label="c1")
ax1.plot(iter_axis, c2_vals, lw=2.0, label="c2")
ax1.set_title("PSO parameter schedules used for Stacking", fontweight="bold")
ax1.set_xlabel("PSO iteration k")
ax1.set_ylabel("Parameter value")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True, edgecolor="black", fancybox=False)
fig.tight_layout()
fig.savefig(PSO_SCHEDULE_PNG, dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

outer_split_audit = pd.DataFrame([
    {
        "Partition": "Train",
        "Samples": len(train_df),
        "Unique_Source_IDs": len(train_sources),
        "Source_IDs": "; ".join(sorted(train_sources)),
    },
    {
        "Partition": "Source-wise Test",
        "Samples": len(test_df),
        "Unique_Source_IDs": len(test_sources),
        "Source_IDs": "; ".join(sorted(test_sources)),
    },
    {
        "Partition": "Independent OOD",
        "Samples": len(df_ood),
        "Unique_Source_IDs": int(df_ood["_source_norm"].nunique()),
        "Source_IDs": "; ".join(sorted(set(df_ood["_source_norm"]))),
    },
])

source_overlap_audit = pd.DataFrame({
    "Check": ["Development vs OOD normalized Source_ID overlap"],
    "Overlap_count": [len(source_overlap)],
    "Overlapping_Source_IDs": ["; ".join(source_overlap)],
})

best_params_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in best_lgbm_params.items()])
config = {
    "script": Path(__file__).name,
    "seed": SEED,
    "development_file": str(CLEAN_FILE_PATH),
    "ood_file": str(OOD_FILE_PATH),
    "target": TARGET,
    "features": FEATURES,
    "group_column": GROUP_COL,
    "outer_split": "GroupShuffleSplit(test_size=0.20, random_state=42) grouped by normalized Source_ID",
    "inner_cv": f"GroupKFold(n_splits={n_splits}) grouped by normalized Source_ID",
    "source_id_overlap_development_ood": len(source_overlap),
    "pso_particles": PSO_PARTICLES,
    "pso_iterations": PSO_ITERATIONS,
    "pso_k_max": K_MAX,
    "pso_schedule": {"W_INI": W_INI, "W_END": W_END, "D1": D1, "D2": D2,
                     "c1_initial": C1_INI, "c1_final": C1_END, "c2_initial": C2_INI, "c2_final": C2_END},
    "pso_objective": "Mean five-fold MSE in fold-specific Yeo-Johnson-transformed target space",
    "lgbm_fixed_n_estimators": N_ESTIMATORS_FIXED,
    "best_lgbm_parameters": best_lgbm_params,
    "dnn_settings": {"epochs": DNN_EPOCHS, "batch_size": DNN_BATCH_SIZE, "first_layer": DNN_N1,
                      "dropout": DNN_DROPOUT, "second_layer": DNN_SECOND_LAYER,
                      "third_layer": DNN_THIRD_LAYER, "learning_rate": DNN_LEARNING_RATE},
    "xgboost_settings": {"n_estimators": XGB_N_ESTIMATORS, "learning_rate": XGB_LEARNING_RATE,
                          "max_depth": XGB_MAX_DEPTH},
    "meta_model": "LinearRegression fitted on source-wise OOF base predictions (DNN, LightGBM, XGBoost)",
    "target_filtering": "None; all valid rows retained including fracture load > 60 kN",
    "ood_in_training_or_model_selection": False,
    "training_metric_note": "Meta-model is fitted to the OOF base feature matrix; Train metrics are diagnostic, not an unbiased estimate.",
    "objective_evaluations_unique": objective_evaluations,
}
with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

final_dnn.save(DNN_MODEL_PATH)
joblib.dump({
    "x_scaler": x_scaler_final,
    "y_scaler": y_scaler_final,
    "lgbm_model": final_lgbm,
    "xgb_model": final_xgb,
    "meta_model": meta_model,
    "best_lgbm_params": best_lgbm_params,
    "feature_names": FEATURES,
    "target_name": TARGET,
    "base_prediction_order": ["DNN", "LightGBM", "XGBoost"],
    "dnn_model_path": str(DNN_MODEL_PATH),
}, MODEL_BUNDLE_PATH)
