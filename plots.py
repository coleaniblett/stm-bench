# plots.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ORDER = ["belief","no-threat","indirect-threat","direct-threat","reward"]

def ensure_figs_dir():
    Path("results/figures").mkdir(parents=True, exist_ok=True)

def pick_latest_scores_base() -> str | None:
    scores_dir = Path("results/scores")
    if not scores_dir.exists(): return None
    cand = sorted(scores_dir.glob("*_pressure.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand: return None
    return cand[0].stem.removesuffix("_pressure")

def to_numeric(df):
    for c in ["n","honesty_rate","compliance_rate","delta_honesty_vs_belief","evasion_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def barplot(df, metric, outdir, order, fname=None):
    cols = [c for c in order if c in df["frame"].unique()]
    pivot = (df.pivot(index="model", columns="frame", values=metric)
               .reindex(columns=cols)
               .fillna(0)
               .sort_index())
    ax = pivot.plot(kind="bar", figsize=(8,4))
    ax.set_ylabel(metric.replace("_"," "))
    ax.set_xlabel("")
    plt.tight_layout()
    out = Path(outdir) / (fname or f"{metric}.png")
    plt.savefig(out, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="(optional) scores base name (e.g., run_1234); defaults to latest *_pressure.csv")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    ensure_figs_dir()
    base = args.base or pick_latest_scores_base()
    if not base:
        raise SystemExit("No scores found in results/scores. Run scorer.py first.")

    press_path = Path("results/scores") / f"{base}_pressure.csv"
    bel_path   = Path("results/scores") / f"{base}_belief.csv"

    press = pd.read_csv(press_path)
    press = to_numeric(press)
    press["frame"] = pd.Categorical(press["frame"], categories=ORDER, ordered=True)

    # honesty plot: include belief row if present (only columns that exist post-scorer change)
    if bel_path.exists():
        bel = pd.read_csv(bel_path)
        bel = to_numeric(bel)
        bel["frame"] = "belief"
        # fill missing metrics so concat aligns
        bel["compliance_rate"] = float("nan")
        bel["delta_honesty_vs_belief"] = 0.0
        bel["evasion_rate"] = float("nan")
        base_cols = ["model","frame","n","honesty_rate","compliance_rate","delta_honesty_vs_belief","evasion_rate"]
        honesty_df = pd.concat([bel[base_cols], press[base_cols]], ignore_index=True)
    else:
        honesty_df = press.copy()

    honesty_df["frame"] = pd.Categorical(honesty_df["frame"], categories=ORDER, ordered=True)
    outdir = args.outdir

    barplot(honesty_df, "honesty_rate", outdir, ORDER)
    if "compliance_rate" in press.columns:         barplot(press, "compliance_rate", outdir, ORDER[1:])
    if "delta_honesty_vs_belief" in press.columns: barplot(press, "delta_honesty_vs_belief", outdir, ORDER[1:])
    if "evasion_rate" in press.columns:            barplot(press, "evasion_rate", outdir, ORDER[1:])

    print(f"Wrote figures to {outdir}")

if __name__ == "__main__":
    main()
