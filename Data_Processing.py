
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PowerTransformer


DATA_FILE = Path(__file__).with_name()
TARGET_COLUMN = "fracture load(KN)"
FEATURE_COLUMNS = [
    "thickness(mm)",
    "Toughness(MJ/m3)",
    "JIc(J/m2)",
    "KI(pa.mm0.5)",
    "KII(pa.mm0.5)",
    "T-stress (Mpa)",
]
TARGET_MAX = 60


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 1.5,
        }
    )


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_excel(path)
    required_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    return data.loc[data[TARGET_COLUMN] <= TARGET_MAX].copy()


def plot_feature_distributions(
    original_data: pd.DataFrame,
    transformed_data,
    transformer: PowerTransformer,
) -> None:
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        figure, (transformed_axis, original_axis) = plt.subplots(
            1, 2, figsize=(11, 5)
        )

        sns.histplot(
            transformed_data[:, index],
            bins=10,
            color="#FF4B4B",
            alpha=0.8,
            edgecolor="black",
            ax=transformed_axis,
        )
        transformed_axis.set_title(
            f"Yeo-Johnson Transformed ($\\lambda$={transformer.lambdas_[index]:.2f})",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        transformed_axis.set_xlabel(f"Transformed {feature_name}")
        transformed_axis.set_ylabel("Count")
        transformed_axis.text(
            0.5,
            -0.25,
            "(b)",
            transform=transformed_axis.transAxes,
            ha="center",
            fontsize=16,
        )

        sns.histplot(
            original_data[feature_name],
            bins=10,
            color="#4A69FF",
            alpha=0.8,
            edgecolor="black",
            ax=original_axis,
        )
        original_axis.set_title(
            "Original Data",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        original_axis.set_xlabel(feature_name)
        original_axis.set_ylabel("Count")
        original_axis.text(
            0.5,
            -0.25,
            "(a)",
            transform=original_axis.transAxes,
            ha="center",
            fontsize=16,
        )

        figure.tight_layout()
        figure.subplots_adjust(bottom=0.2)
        plt.show()
        plt.close(figure)


def main() -> None:
    warnings.filterwarnings("ignore")
    configure_plot_style()

    try:
        data = load_data(DATA_FILE)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Could not load data: {error}") from error

    features = data[FEATURE_COLUMNS]
    transformer = PowerTransformer(method="yeo-johnson")
    transformed_features = transformer.fit_transform(features)

    print(f"Loaded {len(data)} rows from {DATA_FILE.name}.")
    plot_feature_distributions(data, transformed_features, transformer)


if __name__ == "__main__":
    main()
