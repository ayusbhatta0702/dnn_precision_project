"""
Scans outputs/*/precision_table_*.csv (written by train_and_collect.py and
inference_and_collect.py) and builds one consolidated Markdown report:
outputs/FINAL_REPORT.md

Usage:
    python3 src/analyze_report.py --out outputs
Run this last, after train_and_collect.py and the inference_and_collect.py
runs you care about have already produced their outputs/<run>/ folders.
"""
import argparse
import glob
import os

import pandas as pd

FORMATS = ["FP16", "FP8_E4M3", "FP8_E5M2", "INT16", "INT8"]


def load_all(out_root):
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(out_root, "*", "precision_table_*.csv"))):
        run_name = os.path.basename(os.path.dirname(csv_path))
        stage = os.path.basename(csv_path).replace("precision_table_", "").replace(".csv", "")
        df = pd.read_csv(csv_path)
        df["run"] = run_name
        df["stage_file"] = stage
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def verdict_counts_table(df, tensor_type):
    sub = df[df["tensor_type"] == tensor_type]
    if sub.empty:
        return None
    table = (sub.groupby(["format", "verdict"]).size().unstack(fill_value=0))
    table = table.reindex(FORMATS).dropna(how="all")
    return table


def range_summary(df, tensor_type):
    sub = df[(df["tensor_type"] == tensor_type) & (df["format"] == "FP16")]
    if sub.empty:
        return None
    return {
        "n_layers": sub["layer"].nunique(),
        "global_min": sub["min"].min(),
        "global_max": sub["max"].max(),
        "smallest_abs_nonzero": sub["min_abs_nonzero"].min(),
        "median_empirical_spacing": sub["empirical_min_spacing"].median(),
        "max_frac_zero": sub["frac_zero"].max(),
    }


def md_table(df):
    if df is None:
        return "_no data collected for this tensor type_\n"
    return df.to_markdown() + "\n"


def write_report(df, out_path):
    lines = []
    lines.append("# DNN FP32 Range/Precision Analysis — Consolidated Report\n")
    lines.append(f"Runs included: {sorted(df['run'].unique())}\n")

    tensor_types = [t for t in [
        "input", "weight", "activation", "weight_gradient", "weight_update",
        "bias", "logit",
    ] if t in df["tensor_type"].unique()]

    lines.append("\n## 1. Range / precision summary by tensor type\n")
    lines.append("Aggregated across every layer and run that collected this tensor type. "
                  "`smallest_abs_nonzero` and `median_empirical_spacing` are computed from "
                  "the reservoir samples (see README, section 6).\n")
    for tt in tensor_types:
        rs = range_summary(df, tt)
        lines.append(f"\n### {tt}\n")
        if rs is None:
            lines.append("_no data_\n")
            continue
        lines.append(f"- layers/tensors covered: **{rs['n_layers']}**\n")
        lines.append(f"- global min: **{rs['global_min']:.6g}**   global max: **{rs['global_max']:.6g}**\n")
        lines.append(f"- smallest |nonzero| observed: **{rs['smallest_abs_nonzero']:.6g}**\n")
        lines.append(f"- median empirical spacing between distinct sampled values: "
                      f"**{rs['median_empirical_spacing']:.6g}**\n")
        lines.append(f"- largest fraction-of-zeros seen in any single layer: "
                      f"**{rs['max_frac_zero']:.4f}**\n")

    lines.append("\n## 2. FP16 / FP8 / INT16 / INT8 verdict counts by tensor type\n")
    lines.append("Each cell = number of (layer × run) instances of this tensor type that "
                  "received that verdict for that format. See `precision_utils.py` for the "
                  "exact thresholds.\n")
    for tt in tensor_types:
        lines.append(f"\n### {tt}\n")
        lines.append(md_table(verdict_counts_table(df, tt)))

    lines.append("\n## 3. Per-run detail\n")
    for run in sorted(df["run"].unique()):
        rsub = df[df["run"] == run]
        lines.append(f"\n### {run}\n")
        lines.append(f"stages/tensor types collected: "
                      f"{sorted(rsub['stage'].unique())} / {sorted(rsub['tensor_type'].unique())}\n")
        lines.append(f"Full per-layer numbers: `outputs/{run}/precision_table_*.csv`  \n")
        lines.append(f"Plots: `outputs/{run}/plots/<stage>/<layer>/<tensor_type>_{{full,zoom}}.png`\n")

    lines.append("\n## 4. Discussion (fill in after reviewing the numbers/plots above)\n")
    lines.append(_discussion_template())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _discussion_template():
    return """
Use the tables and plots above to answer, for each tensor type:

- **Weights**: typically small-magnitude (|w| usually < 1), roughly
  bell-shaped, no long tail → look at the FP16/INT8 verdict counts: if
  almost all layers show "Sufficient" for FP16 and INT8 (with per-tensor
  scaling), weights are a strong candidate for low precision, which matches
  common quantization practice (post-training INT8 weight quantization is
  standard).
- **Activations**: range varies a lot by layer (pre-ReLU vs. post-ReLU,
  batch-norm outputs, pooled features) and can have long tails / large
  values right after a conv+BN with no normalization following it. Check
  whether INT8's `frac_underflow_of_nonzero` / `dynamic_range_ratio`
  columns blow up for any layer — that signals INT8 needs per-layer (not
  global) scaling, or FP8/FP16 instead.
- **Gradients / weight updates**: usually have the widest dynamic range of
  all (many near-zero values alongside occasional large spikes, especially
  early in training) — expect INT8 (and often INT16) to show "Marginal"/
  "Not sufficient" here due to `dynamic_range_ratio`; this matches why
  mixed-precision training keeps a FP32 (or FP16 with loss-scaling) master
  copy of gradients/weight-updates even when forward-pass activations run
  in FP16/INT8.
- **Biases**: usually low-dimensional and small-magnitude per layer, close
  to weights in behavior — check verdicts the same way as weights.
- **Output logits**: pre-softmax values can have a moderate range; check
  whether FP16 is sufficient (commonly yes) since logits typically don't
  need the wide dynamic range gradients do.
- **Across training stages** (begin vs. middle vs. end): compare the same
  layer's gradient/weight-update verdicts at the three stages — gradients
  are usually largest and most spread out at the *beginning* of training
  and shrink as the model converges, so low-precision gradient
  representations are often hardest to justify early in training.

Fill in the specific layer names / numbers from section 2/3 above to turn
this into your final written conclusion.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs", help="root outputs dir to scan and write FINAL_REPORT.md into")
    args = ap.parse_args()

    df = load_all(args.out)
    if df is None:
        print(f"No precision_table_*.csv files found under {args.out}/*/. "
              f"Run train_and_collect.py / inference_and_collect.py first.")
        return

    report_path = os.path.join(args.out, "FINAL_REPORT.md")
    write_report(df, report_path)
    print(f"Wrote {report_path}  ({len(df)} rows from {df['run'].nunique()} run(s))")


if __name__ == "__main__":
    main()
