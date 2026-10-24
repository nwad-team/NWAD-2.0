
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from load_data import load_dataset


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

    # The source column is "Label" (capital L); resolve it case-insensitively.
    label_column = next(
        (c for c in df.columns if c.lower() == "label"), None
    )
    if label_column is None:
        logging.warning("'label' column not found; defaulting to attack_cat only.")
        benign = pd.Series(False, index=df.index)
    else:
        benign = pd.to_numeric(df[label_column], errors="coerce").eq(0)

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
# Text summary (markdown report)
# Each plot function appends a human- and machine-readable
# section here so the figures can be reviewed without opening
# the images.
# =========================================================

REPORT_LINES: list[str] = []


def _add_report(*lines: str) -> None:
    REPORT_LINES.extend(lines)


def _md_table(frame: pd.DataFrame, float_format: str = "{:.3f}") -> None:
    """Append a pandas frame as a GitHub-flavored markdown table."""
    if frame is None or frame.empty:
        _add_report("(no data)")
        return
    # Only include the index as a column when it carries meaningful
    # labels (e.g. class names), not the default 0..n-1 range.
    index_is_default = (
        isinstance(frame.index, pd.RangeIndex)
        and frame.index.name is None
        and frame.index.start == 0
        and frame.index.step == 1
    )

    # When the index carries meaningful labels (e.g. class names), it is
    # rendered as a leading column, so the header and separator must
    # include it too or the markdown table misaligns.
    if index_is_default:
        header_cells = [str(c) for c in frame.columns]
    else:
        header_cells = [
            "" if frame.index.name is None else str(frame.index.name)
        ] + [str(c) for c in frame.columns]

    _add_report("| " + " | ".join(header_cells) + " |")
    _add_report("| " + " | ".join(["---"] * len(header_cells)) + " |")

    for idx, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(float_format.format(value))
            else:
                cells.append(str(value))
        prefix = "" if index_is_default else f"{idx} | "
        _add_report("| " + prefix + " | ".join(cells) + " |")


def write_report() -> None:
    path = OUTPUT_DIR / "feature_analysis_summary.md"
    body = "\n".join(REPORT_LINES).rstrip()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body + "\n")
    logging.info("Saved text summary %s", path)


FEATURE_DESCRIPTIONS: dict[str, str] = {}


def load_feature_descriptions() -> dict[str, str]:
    """Read the human-readable feature descriptions from the raw dataset
    documentation (NUSW-NB15_features.csv) into a name -> description map.

    Names are lowercased so lookups are case-insensitive (the data mixes
    cases, e.g. "Spkts"). The result is cached on first load.
    """
    if FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS

    features_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "raw" / "NUSW-NB15_features.csv"
    )
    if not features_path.is_file():
        logging.warning("Feature description file not found: %s", features_path)
        return FEATURE_DESCRIPTIONS

    # The features CSV is Windows-1252 encoded (e.g. curly quotes as 0x92),
    # so decode with cp1252 to preserve the intended characters.
    try:
        feats = pd.read_csv(features_path, encoding="cp1252")
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Could not read feature descriptions: %s", exc)
        return FEATURE_DESCRIPTIONS

    # Column names carry stray whitespace ("Type "); normalize them.
    feats.columns = [c.strip().lower() for c in feats.columns]
    if "name" in feats.columns and "description" in feats.columns:
        for name, desc in zip(feats["name"], feats["description"]):
            if pd.isna(name) or pd.isna(desc):
                continue
            FEATURE_DESCRIPTIONS[str(name).strip().lower()] = str(desc).strip()

    return FEATURE_DESCRIPTIONS


def describe(name: str) -> str:
    """Return the documented description for a feature name ('' if unknown)."""
    load_feature_descriptions()  # ensure the map is populated (no-op if loaded)
    return FEATURE_DESCRIPTIONS.get(str(name).strip().lower(), "")


# =========================================================
# 1. Class distribution
# =========================================================

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["class_name"].value_counts().sort_values()

    # --- Text summary ---
    total = int(counts.sum())
    counts_frame = counts.to_frame("flows").sort_values("flows", ascending=False)
    counts_frame["% of flows"] = (100.0 * counts_frame["flows"] / total).round(2)

    _add_report(
        "## 1. Class distribution",
        "",
        f"Total flows: **{total:,}** across **{len(counts)}** classes.",
        "",
    )
    _md_table(counts_frame, float_format="{:.2f}")
    _add_report("")

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
    # The source column is "Stime" (capital S); resolve it case-insensitively.
    stime_column = next(
        (c for c in df.columns if c.lower() == "stime"), None
    )
    if stime_column is None:
        logging.warning("stime not found.")
        return

    # UNSW-NB15 commonly stores stime as Unix seconds.
    timestamps = pd.to_numeric(df[stime_column], errors="coerce")

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

    # --- Text summary ---
    _add_report(
        "## 2. Class distribution over time (30-minute bins)",
        "",
        f"Time range: **{counts.index.min()}** to **{counts.index.max()}** "
        f"({len(counts)} bins).",
        "",
        "Total flows per class:",
        "",
    )
    _md_table(counts.sum(axis=0).sort_values(ascending=False).to_frame("total"),
              float_format="{:,.0f}")
    _add_report("")

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

    # --- Text summary: for each class, the bin where it peaks ---
    _add_report(
        "Peak share of each class within a 30-min bin:",
        "",
    )
    peak_rows = []
    for class_name in proportions.columns:
        series = proportions[class_name].dropna()
        if series.empty:
            continue
        peak_bin = series.idxmax()
        peak_rows.append((class_name, f"{peak_bin:%Y-%m-%d %H:%M}", f"{series.max():.2%}"))
    _md_table(pd.DataFrame(peak_rows, columns=["class", "peak bin", "peak share"]),
              float_format="{:.2%}")
    _add_report("")

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

        # --- Text summary: top ports, dominant class, and share ---
        port_totals = counts.sum(axis=1).sort_values(ascending=False)
        port_total_sum = int(port_totals.sum())
        summary_rows = []
        for port in port_totals.index:
            row = counts.loc[port]
            row_total = int(row.sum())
            dominant_class = row.idxmax()
            dominant_share = (row.max() / row_total) if row_total else 0.0
            summary_rows.append((
                port,
                f"{int(port_totals[port]):,}",
                f"{port_totals[port] / port_total_sum:.2%}" if port_total_sum else "0.00%",
                dominant_class,
                f"{dominant_share:.2%}",
            ))
        _add_report(
            f"### Top {len(summary_rows)} `{column}` ports by class",
            "",
        )
        _md_table(
            pd.DataFrame(
                summary_rows,
                columns=["port", "flows", "share", "dominant class", "dominant share"],
            )
        )
        _add_report("")

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
    # Compare case-insensitively: the source data mixes cases (e.g. "Stime").
    excluded = {
        "label", "attack_cat", "class_name",
        "srcip", "dstip", "sport", "dsport",
        "stime", "ltime",
    }

    numeric_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column.lower() not in excluded
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

    # --- Text summary: strongest off-diagonal correlations ---
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    stacked = tri.stack().dropna()
    order = stacked.abs().sort_values(ascending=False).head(15).index

    _add_report(
        "## 4. Strongest feature correlations (Spearman, off-diagonal)",
        "",
        f"Features in matrix: **{len(corr.columns)}**. Top 15 by |corr| "
        "(with each feature's documented description):",
        "",
    )
    for a, b in order:
        corr_value = stacked.loc[(a, b)]
        _add_report(
            f"**{a} <-> {b}** — {corr_value:.3f}",
            f"- `{a}`: {describe(a) or '(no description found)'}",
            f"- `{b}`: {describe(b) or '(no description found)'}",
            "",
        )


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

    # --- Text summary: descriptive stats for the features plotted ---
    _add_report(
        "## 5. Scatter-matrix features (descriptive statistics)",
        "",
        f"Features plotted: **{', '.join(features)}** "
        f"(log1p-transformed if nonnegative).",
        "",
    )
    stats = sample[features].describe().T[
        ["count", "mean", "std", "min", "50%", "max"]
    ]
    _md_table(stats, float_format="{:,.4f}")
    _add_report("")

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
    # Start from a clean report each run (the notebook kernel keeps the module
    # alive, so without this, re-running main() would append duplicate sections).
    REPORT_LINES.clear()

    logging.info("Loading dataset...")
    df = load_dataset()
    df = prepare_data(df)

    logging.info("Dataset shape: %s", df.shape)

    _add_report(
        "# UNSW-NB15 Feature Analysis - Text Summary",
        "",
        f"Dataset shape: **{df.shape[0]:,}** flows x **{df.shape[1]}** columns. "
        f"Generated by `src/analyze_features.py`.",
        "",
        "This file summarizes the accompanying PNG plots in text so they "
        "can be reviewed without opening the images.",
        "",
    )

    plot_class_distribution(df)
    plot_classes_over_time(df)
    plot_ports_by_class(df)
    plot_correlation_matrix(df)
    plot_scatter_matrix(df)

    write_report()

    logging.info("All plots saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
