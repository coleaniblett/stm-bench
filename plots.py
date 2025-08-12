# plots.py
import argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt

ORDER = ["belief", "no-threat", "indirect-threat", "direct-threat", "reward"]

def to_numeric(df):
    for c in ["n","honesty_rate","compliance_rate","dishonest_compliance_rate","delta_honesty_vs_belief"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def barplot(df, metric, outdir, order):
    cols = [c for c in order if c in df.columns or c in df.get("frame",[]).unique()]
    pivot = (df.pivot(index="model", columns="frame", values=metric)
               .reindex(columns=[c for c in order if c in df["frame"].unique()])
               .fillna(0)
               .sort_index())
    ax = pivot.plot(kind="bar", figsize=(8,4))
    ax.set_ylabel(metric.replace("_"," "))
    ax.set_xlabel("")
    plt.tight_layout()
    out = pathlib.Path(outdir) / f"{metric}.png"
    plt.savefig(out, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pressure", required=True, help="results/scores/<run>_pressure.csv")
    ap.add_argument("--belief", help="results/scores/<run>_belief.csv (optional)")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load summaries
    press = pd.read_csv(args.pressure)
    press = to_numeric(press)
    press["frame"] = pd.Categorical(press["frame"], categories=ORDER, ordered=True)

    # Combine belief for honesty plot (optional)
    if args.belief:
        bel = pd.read_csv(args.belief)
        bel = to_numeric(bel)
        bel["frame"] = "belief"
        bel["compliance_rate"] = float("nan")
        bel["dishonest_compliance_rate"] = float("nan")
        bel["delta_honesty_vs_belief"] = 0.0
        bel = bel[["model","frame","n","honesty_rate","compliance_rate","dishonest_compliance_rate","delta_honesty_vs_belief"]]
        honesty_df = pd.concat([bel, press[bel.columns]], ignore_index=True)
    else:
        honesty_df = press.copy()

    honesty_df["frame"] = pd.Categorical(honesty_df["frame"], categories=ORDER, ordered=True)

    # 1) honesty_rate (includes belief if provided)
    barplot(honesty_df, "honesty_rate", outdir, ORDER)

    # 2) compliance_rate (pressure only)
    if "compliance_rate" in press.columns:
        barplot(press, "compliance_rate", outdir, ORDER[1:])  # exclude belief

    # 3) dishonest_compliance_rate (pressure only)
    if "dishonest_compliance_rate" in press.columns:
        barplot(press, "dishonest_compliance_rate", outdir, ORDER[1:])

    # 4) delta_honesty_vs_belief (already in pressure)
    if "delta_honesty_vs_belief" in press.columns:
        barplot(press, "delta_honesty_vs_belief", outdir, ORDER[1:])

if __name__ == "__main__":
    main()
