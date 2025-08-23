# plots.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_pressure_stack_by_frame(dist_df: pd.DataFrame, outname: str = "pressure_honesty_stack_by_frame.png"):
    """
    Stacked bar: for each (model, frame) on x, three segments stacked to 100%:
      - Honest (green), Dishonest (red), Evasive (orange)
    """
    if dist_df.empty:
        raise SystemExit("No pressure-by-frame distribution data to plot.")

    x_labels = dist_df["run_frame_label"].tolist()
    x = range(len(x_labels))

    honest = dist_df["pct_honest"].values
    dishonest = dist_df["pct_dishonest"].values
    evasive = dist_df["pct_evasive"].values

    plt.figure(figsize=(max(10, len(x_labels)*1.1), 5))
    b1 = plt.bar(x, honest, label="Honest", color="green")
    b2 = plt.bar(x, dishonest, bottom=honest, label="Dishonest", color="red")
    b3 = plt.bar(x, evasive, bottom=honest + dishonest, label="Evasive", color="blue")

    plt.xticks(ticks=x, labels=x_labels, rotation=30, ha="right")
    plt.ylabel("Percent of responses under pressure")
    plt.title("Honest vs Dishonest vs Evasive by Pressure Scenario")
    plt.ylim(0, 100)

    # Legend order matches stack order
    plt.legend((b3[0], b2[0], b1[0]), ("Evasive", "Dishonest", "Honest"), loc="upper right")

    plt.tight_layout()
    outpath = FIGS_DIR / outname
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Wrote {outpath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",
                        help="Optional: a specific run base or path to a *_scored.csv. "
                             "If omitted, all results/scores/*_scored.csv are included.")
    parser.add_argument("--outdir", default=str(FIGS_DIR))
    args = parser.parse_args()

    ensure_dirs()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    paths = find_scored_paths(args.base)

    dists = []
    for p in paths:
        df = pd.read_csv(p)
        dists.append(build_pressure_distribution_by_frame(df))
    dist_df = pd.concat(dists, ignore_index=True)

    plot_pressure_stack_by_frame(dist_df, outname="pressure_honesty_stack_by_frame.png")

if __name__ == "__main__":
    main()
