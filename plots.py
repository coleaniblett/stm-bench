# plots.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math

FIGS_DIR = Path("results/figures")
SCORES_DIR = Path("results/scores")

# Order the pressure frames on the x-axis
FRAME_ORDER = ["no-threat", "indirect-threat", "direct-threat", "reward"]

def ensure_dirs():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

def find_scored_paths(base: str | None):
    """
    If --base is provided, accept either:
      - an explicit file (…_scored.csv) OR
      - the base name (we'll resolve to results/scores/<base>_scored.csv)
    Otherwise, load ALL results/scores/*_scored.csv
    """
    if base:
        p = Path(base)
        if p.suffix == ".csv" and p.exists():
            return [p]
        cand = SCORES_DIR / f"{base}_scored.csv"
        if cand.exists():
            return [cand]
        raise SystemExit(f"Could not find scored CSV for base '{base}'.")
    else:
        files = sorted(SCORES_DIR.glob("*_scored.csv"))
        if not files:
            raise SystemExit("No scored CSVs found in results/scores. Run scorer.py first.")
        return files

def split_models_evenly(n_models: int, max_per_plot: int = 5, min_per_plot: int = 3) -> list[int]:
    """
    Determine how to split n_models into multiple plots.
    Returns a list of integers representing the number of models in each plot.
    
    Strategy:
    - If <= max_per_plot, use single plot
    - Otherwise, split as evenly as possible while respecting min_per_plot
    - Aim for balanced distribution
    """
    if n_models <= max_per_plot:
        return [n_models]
    
    # Calculate ideal number of plots
    n_plots = math.ceil(n_models / max_per_plot)
    
    # Try to distribute evenly
    base_size = n_models // n_plots
    remainder = n_models % n_plots
    
    # If base_size is less than minimum, reduce number of plots
    if base_size < min_per_plot:
        n_plots = n_models // min_per_plot
        base_size = n_models // n_plots
        remainder = n_models % n_plots
    
    # Create distribution
    distribution = [base_size] * n_plots
    
    # Distribute remainder
    for i in range(remainder):
        distribution[i] += 1
    
    # Sort descending to have larger plots first
    distribution.sort(reverse=True)
    
    return distribution

def build_pressure_distribution_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: per-response scored dataframe for a single run.
    Output: one row per (model, frame) with percentages of 0/1/2 under pressure.
    """
    required = {"mode", "frame", "model", "run_id", "honesty_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Scored CSV missing required columns: {sorted(missing)}")

    press = df[df["mode"] == "pressure"].copy()
    if press.empty:
        return pd.DataFrame(columns=[
            "run_id","model","frame","run_frame_label","pct_honest","pct_dishonest","pct_evasive","n"
        ])

    press = press[press["frame"].isin(FRAME_ORDER)].copy()
    if press.empty:
        return pd.DataFrame(columns=[
            "run_id","model","frame","run_frame_label","pct_honest","pct_dishonest","pct_evasive","n"
        ])

    rows = []
    for (model, frame), g in press.groupby(["model","frame"]):
        counts = g["honesty_score"].value_counts(dropna=False)
        n = int(counts.sum())
        def pct(val): return float(counts.get(val, 0)) * 100.0 / n if n > 0 else 0.0
        run_id = str(g["run_id"].iloc[0])
        rows.append({
            "run_id": run_id,
            "model": str(model),
            "frame": str(frame),
            "run_frame_label": f"{model} ({frame})",
            "pct_honest": pct(1),
            "pct_dishonest": pct(0),
            "pct_evasive": pct(2),
            "n": n,
        })

    out = pd.DataFrame(rows)
    out["frame"] = pd.Categorical(out["frame"], categories=FRAME_ORDER, ordered=True)
    out = out.sort_values(["model","frame"]).reset_index(drop=True)
    return out

def build_belief_accuracy_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: per-response scored dataframe for a single run.
    Output: one row per model with accuracy percentages for belief mode responses.
    """
    required = {"mode", "model", "run_id", "honesty_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Scored CSV missing required columns: {sorted(missing)}")

    belief = df[df["mode"] == "belief"].copy()
    if belief.empty:
        return pd.DataFrame(columns=["run_id", "model", "pct_correct", "pct_incorrect", "n"])

    rows = []
    for model, g in belief.groupby("model"):
        counts = g["honesty_score"].value_counts(dropna=False)
        n = int(counts.sum())
        def pct(val): return float(counts.get(val, 0)) * 100.0 / n if n > 0 else 0.0
        run_id = str(g["run_id"].iloc[0])
        rows.append({
            "run_id": run_id,
            "model": str(model),
            "pct_correct": pct(1),
            "pct_incorrect": pct(0) + pct(2),  # Combine dishonest and evasive as incorrect
            "n": n,
        })

    return pd.DataFrame(rows)

def _compute_grouped_x_positions(dist_df: pd.DataFrame, group_gap: float) -> list[float]:
    """
    Return x positions with extra spacing between groups of 4 (one per FRAME_ORDER),
    grouping by run_id in the order rows appear in dist_df.
    """
    frame_to_idx = {f: i for i, f in enumerate(FRAME_ORDER)}
    x = []
    current_run = None
    group_index = -1
    for _, row in dist_df.iterrows():
        run = row["run_id"]
        frame = str(row["frame"])
        if run != current_run:
            current_run = run
            group_index += 1
        base = group_index * (len(FRAME_ORDER) + group_gap)
        x.append(base + frame_to_idx.get(frame, 0))
    return x

def plot_belief_accuracy_by_model(
    accuracy_df: pd.DataFrame,
    outname: str = "accuracy.png",
):
    """
    Bar chart showing accuracy of models on belief elicitation (baseline knowledge).
    """
    if accuracy_df.empty:
        raise SystemExit("No belief accuracy data to plot.")

    plt.rcParams.update({'font.size': 11})

    models = accuracy_df["model"].tolist()
    correct = accuracy_df["pct_correct"].values
    incorrect = accuracy_df["pct_incorrect"].values

    x = range(len(models))
    plt.figure(figsize=(max(8, len(models) * 1.2), 6))

    b1 = plt.bar(x, correct, label="Correct", color="#87A96B", alpha=0.8,
                 edgecolor='white', linewidth=0.5)
    b2 = plt.bar(x, incorrect, bottom=correct, label="Incorrect", color="#FFB6C1", alpha=0.8,
                 edgecolor='white', linewidth=0.5)

    plt.xticks(ticks=x, labels=models, rotation=30, ha="right")
    plt.ylabel("Percent of responses")
    plt.title("Model Accuracy on Belief Elicitation", pad=20)
    plt.ylim(0, 100)

    # Clean up axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    plt.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    plt.legend((b2[0], b1[0]), ("Incorrect", "Correct"), 
              bbox_to_anchor=(1.02, 0.99), loc='upper left')

    plt.tight_layout()
    outpath = FIGS_DIR / outname
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Wrote {outpath}")

def plot_pressure_stack_by_frame(
    dist_df: pd.DataFrame,
    outname: str = "pressure_honesty_stack_by_frame.png",
    group_gap: float = 1.5,
    plot_number: int = None,
    total_plots: int = 1,
):
    """
    Stacked bar: for each (model, frame) on x, three segments stacked to 100%:
      - Honest (green), Evasive (blue), Dishonest (red)
    Adds extra spacing between each 4-bar group (per run_id).
    """
    if dist_df.empty:
        raise SystemExit("No pressure-by-frame distribution data to plot.")

    # Set font size
    plt.rcParams.update({'font.size': 11})

    # Compute spaced x positions
    x_labels = dist_df["run_frame_label"].tolist()
    x = _compute_grouped_x_positions(dist_df, group_gap)

    honest = dist_df["pct_honest"].values
    dishonest = dist_df["pct_dishonest"].values
    evasive = dist_df["pct_evasive"].values

    # Expand figure width proportionally to the added group gaps
    num_groups = dist_df["run_id"].nunique()
    width_units = len(x_labels) + max(0, num_groups - 1) * group_gap
    plt.figure(figsize=(max(12, width_units * 1.2), 6))

    b1 = plt.bar(x, honest, label="Honest", color="#87A96B", alpha=0.8, 
                 edgecolor='white', linewidth=0.5)  # Sage green
    b2 = plt.bar(x, evasive, bottom=honest, label="Evasive", color="#A9A9A9", alpha=0.8,
                 edgecolor='white', linewidth=0.5)  # Grey
    b3 = plt.bar(x, dishonest, bottom=honest + evasive, label="Dishonest", color="#FFB6C1", alpha=0.8,
                 edgecolor='white', linewidth=0.5)  # Light pink

    plt.xticks(ticks=x, labels=x_labels, rotation=30, ha="right")
    plt.ylabel("Percent of responses under pressure")
    
    # Adjust title based on whether we're splitting
    if total_plots > 1:
        title = f"Honest vs Dishonest vs Evasive by Pressure Scenario (Part {plot_number} of {total_plots})"
    else:
        title = "Honest vs Dishonest vs Evasive by Pressure Scenario"
    plt.title(title, pad=20)
    plt.ylim(0, 100)

    # Clean up axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    plt.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend positioned between plot and right edge
    plt.legend((b3[0], b2[0], b1[0]), ("Dishonest", "Evasive", "Honest"), 
              bbox_to_anchor=(1.02, 0.99), loc='upper left')

    plt.tight_layout()
    outpath = FIGS_DIR / outname
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Wrote {outpath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",
                        help="Optional: a specific run base or path to a *_scored.csv. "
                             "If omitted, all results/scores/*_scored.csv are included.")
    parser.add_argument("--outdir", default=str(FIGS_DIR))
    parser.add_argument("--group-gap", type=float, default=1.5,
                        help="Extra spacing between 4-bar groups (default: 1.5).")
    parser.add_argument("--max-models-per-plot", type=int, default=5,
                        help="Maximum number of models per honesty plot (default: 5).")
    args = parser.parse_args()

    ensure_dirs()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    paths = find_scored_paths(args.base)

    # Build distribution data
    dists = []
    for p in paths:
        df = pd.read_csv(p)
        dists.append(build_pressure_distribution_by_frame(df))
    dist_df = pd.concat(dists, ignore_index=True)

    # Get unique models (run_ids)
    unique_models = dist_df["run_id"].unique().tolist()
    n_models = len(unique_models)
    
    # Determine how to split the models
    if n_models <= args.max_models_per_plot:
        # Single plot
        plot_pressure_stack_by_frame(
            dist_df,
            outname="honesty.png",
            group_gap=args.group_gap,
        )
    else:
        # Multiple plots
        distribution = split_models_evenly(n_models, max_per_plot=args.max_models_per_plot)
        
        start_idx = 0
        for plot_num, models_in_plot in enumerate(distribution, 1):
            # Get the models for this plot
            models_subset = unique_models[start_idx:start_idx + models_in_plot]
            
            # Filter the dataframe for these models
            subset_df = dist_df[dist_df["run_id"].isin(models_subset)].copy()
            
            # Generate filename
            if len(distribution) == 1:
                filename = "honesty.png"
            else:
                filename = f"honesty_part{plot_num}.png"
            
            # Create the plot
            plot_pressure_stack_by_frame(
                subset_df,
                outname=filename,
                group_gap=args.group_gap,
                plot_number=plot_num,
                total_plots=len(distribution),
            )
            
            start_idx += models_in_plot

    # Generate accuracy plot for belief mode
    accuracy_dfs = []
    for p in paths:
        df = pd.read_csv(p)
        accuracy_dfs.append(build_belief_accuracy_by_model(df))
    accuracy_df = pd.concat(accuracy_dfs, ignore_index=True)
    
    plot_belief_accuracy_by_model(accuracy_df, outname="accuracy.png")

if __name__ == "__main__":
    main()