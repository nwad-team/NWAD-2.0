
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from data.data_loading import load_dataset


# =========================================================
# Configuration
# =========================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "dataset"
    / "processed"
    / "feature_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CORRELATION_SAMPLE_SIZE = 100_000
SCATTER_SAMPLE_SIZE = 2_000
TOP_N_PORTS = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


# =========================================================
# Data preparation
# =========================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create a readable label for benign and attack traffic.
    df["class_name"] = df["attack_cat"].astype("string").str.strip()

    benign = pd.to_numeric(
        df["label"], errors="coerce"
    ).eq(0)

    df.loc[benign, "class_name"] = "Benign"
    df["class_name"] = df["class_name"].fillna("UnknownAttack")

    return df


def save_plot(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logging.info("Saved %s", path)


# =========================================================
# 1. Class distribution
# =========================================================

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["class_name"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax)

    ax.set_title("UNSW-NB15: Class Distribution")
    ax.set_xlabel("Number of flows")
    ax.set_ylabel("Traffic class")

    save_plot("01_class_distribution.png")

    # Logarithmic version makes minority classes easier to compare.
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax)

    ax.set_xscale("log")
    ax.set_title("UNSW-NB15: Class Distribution (Log Scale)")
    ax.set_xlabel("Number of flows (log scale)")
    ax.set_ylabel("Traffic class")

    save_plot("02_class_distribution_log.png")


# =========================================================
# 2. Class distribution over time
# =========================================================

def plot_classes_over_time(df: pd.DataFrame) -> None:
    if "stime" not in df.columns:
        logging.warning("stime not found.")
        return

    # UNSW-NB15 commonly stores stime as Unix seconds.
    timestamps = pd.to_numeric(df["stime"], errors="coerce")

    valid = timestamps.notna() & timestamps.gt(0)

    if not valid.any():
        logging.warning("No valid numeric timestamps found.")
        return

    temp = df.loc[valid, ["class_name"]].copy()
    temp["timestamp"] = pd.to_datetime(
        timestamps.loc[valid],
        unit="s",
        errors="coerce",
    )
    temp = temp.dropna(subset=["timestamp"])

    if temp.empty:
        logging.warning("Could not parse timestamps.")
        return

    # Count flows per class in 30-minute intervals.
    temp["time_bin"] = temp["timestamp"].dt.floor("30min")

    counts = (
        temp.groupby(["time_bin", "class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    for class_name in counts.columns:
        ax.plot(
            counts.index,
            counts[class_name],
            label=class_name,
            linewidth=1,
        )

    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("Traffic Class Counts Over Time")
    ax.set_xlabel("Flow start time")
    ax.set_ylabel("Number of flows per 30 minutes")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_plot("03_classes_over_time.png")

    # Normalize each time interval to show composition, rather
    # than allowing Benign to dominate solely by count.
    proportions = counts.div(
        counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    proportions.plot.area(ax=ax, linewidth=0)

    ax.set_title("Traffic Class Proportions Over Time")
    ax.set_xlabel("Flow start time")
    ax.set_ylabel("Proportion of flows")
    ax.set_ylim(0, 1)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_plot("04_class_proportions_over_time.png")


# =========================================================
# 3. Source and destination ports by class
# =========================================================

def plot_ports_by_class(df: pd.DataFrame) -> None:
    for column in ("sport", "dsport"):
        if column not in df.columns:
            logging.warning("%s not found.", column)
            continue

        ports = pd.to_numeric(df[column], errors="coerce")
        valid = ports.between(0, 65535)

        temp = df.loc[valid, ["class_name"]].copy()
        temp["port"] = ports.loc[valid].astype("int64")

        top_ports = temp["port"].value_counts().head(
            TOP_N_PORTS
        ).index

        # Show how each class contributes to each frequent port.
        counts = pd.crosstab(
            temp.loc[temp["port"].isin(top_ports), "port"],
            temp.loc[temp["port"].isin(top_ports), "class_name"],
        )

        counts = counts.loc[
            counts.sum(axis=1).sort_values(ascending=False).index
        ]

        fig, ax = plt.subplots(figsize=(13, 7))
        counts.plot(kind="bar", stacked=True, ax=ax)

        ax.set_title(f"Top {TOP_N_PORTS} {column} Values by Class")
        ax.set_xlabel(column)
        ax.set_ylabel("Number of flows")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

        save_plot(f"05_{column}_counts_by_class.png")

        # Proportion of classes within each port.
        proportions = counts.div(
            counts.sum(axis=1).replace(0, np.nan),
            axis=0,
        )

        fig, ax = plt.subplots(figsize=(13, 7))
        proportions.plot(kind="bar", stacked=True, ax=ax)

        ax.set_title(f"Class Composition for Frequent {column} Values")
        ax.set_xlabel(column)
        ax.set_ylabel("Proportion of flows")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

        save_plot(f"06_{column}_class_proportions.png")

        print(f"\nMost frequent values in {column}:")
        print(temp["port"].value_counts().head(TOP_N_PORTS))


# =========================================================
# 4. Numeric feature correlations
# =========================================================

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    # Exclude labels, identifiers, timestamps, and ports.
    excluded = {
        "label", "attack_cat", "class_name",
        "srcip", "dstip", "sport", "dsport",
        "stime", "ltime",
    }

    numeric_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column not in excluded
    ]

    sample = df.sample(
        n=min(CORRELATION_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )

    numeric_df = sample[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.loc[
        :, numeric_df.nunique(dropna=True) > 1
    ]

    # Spearman detects monotonic relationships, including some
    # nonlinear relationships that Pearson may miss.
    corr = numeric_df.corr(method="spearman", min_periods=100)

    fig_size = max(12, len(corr.columns) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    image = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Spearman Correlation: Numeric Features")

    fig.colorbar(image, ax=ax, label="Spearman correlation")

    save_plot("07_numeric_correlation_matrix.png")

    corr.to_csv(OUTPUT_DIR / "numeric_correlations.csv")


# =========================================================
# 5. Scatter matrix
# =========================================================

def plot_scatter_matrix(df: pd.DataFrame) -> None:
    features = [
        "dur", "sbytes", "dbytes", "spkts", "dpkts",
        "sload", "dload", "sttl", "dttl", "tcprtt",
    ]

    features = [
        col for col in features
        if col in df.columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(features) < 2:
        logging.warning("Not enough numeric features for scatter matrix.")
        return

    sample = df.sample(
        n=min(SCATTER_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )

    plot_df = sample[features].replace(
        [np.inf, -np.inf], np.nan
    )

    # Log transform nonnegative features to reduce skew.
    for column in features:
        if (plot_df[column].dropna() >= 0).all():
            plot_df[column] = np.log1p(plot_df[column])

    plot_df = plot_df.dropna()

    class_codes = (
        sample.loc[plot_df.index, "class_name"]
        .astype("category")
        .cat.codes
    )

    axes = scatter_matrix(
        plot_df,
        figsize=(15, 15),
        diagonal="hist",
        alpha=0.3,
        s=8,
        c=class_codes,
        marker=".",
    )

    plt.suptitle(
        "Scatter Matrix of Selected Numeric Features",
        y=1.01,
    )

    save_plot("08_scatter_matrix.png")


# =========================================================
# Main
# =========================================================

def main() -> None:
    logging.info("Loading dataset...")
    df = load_dataset()
    df = prepare_data(df)

    logging.info("Dataset shape: %s", df.shape)

    plot_class_distribution(df)
    plot_classes_over_time(df)
    plot_ports_by_class(df)
    plot_correlation_matrix(df)
    plot_scatter_matrix(df)

    logging.info("All plots saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
