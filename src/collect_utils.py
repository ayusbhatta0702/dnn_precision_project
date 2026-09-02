"""
Shared postprocessing: given a populated hooks.StatsRegistry, write out
- outputs/<run>/stats_summary.json        (numeric stats only, all layers/types)
- outputs/<run>/plots/<stage>/<layer>/<tensor_type>_{full,zoom}.png
- outputs/<run>/precision_table.csv       (long format: one row per format)

Used identically by train_and_collect.py (per stage) and
inference_and_collect.py (single pass).
"""
import json
import os
import numpy as np
import pandas as pd

from plotting import plot_distribution
from precision_utils import full_precision_analysis


def process_registry(registry, out_dir, stage_label="all", max_sample_out=20_000,
                      make_plots=True, precision_sample_cap=100_000):
    """Returns (stats_summary: dict, precision_rows: list[dict])."""
    plots_dir = os.path.join(out_dir, "plots", stage_label)
    summaries = registry.export_summaries(max_sample_out=max_sample_out)

    stats_summary = {}
    precision_rows = []

    for layer_name, tensor_types in summaries.items():
        stats_summary[layer_name] = {}
        for tensor_type, data in tensor_types.items():
            sample = np.array(data.get("sample", []), dtype=np.float64)
            light = {k: v for k, v in data.items() if k != "sample"}
            stats_summary[layer_name][tensor_type] = light

            if sample.size == 0:
                continue

            if make_plots:
                safe_layer = layer_name.replace(".", "_") or "root"
                prefix = os.path.join(plots_dir, safe_layer, tensor_type)
                plot_distribution(
                    sample,
                    title=f"{layer_name} [{stage_label}] · {tensor_type} "
                          f"({light.get('layer_type', '?')})",
                    out_prefix=prefix,
                    stats=light,
                )

            prec_sample = sample
            if prec_sample.size > precision_sample_cap:
                idx = np.random.default_rng(0).choice(prec_sample.size, precision_sample_cap, replace=False)
                prec_sample = prec_sample[idx]
            prec = full_precision_analysis(prec_sample)

            for fmt_key, fmt_res in prec.items():
                row = {
                    "stage": stage_label,
                    "layer": layer_name,
                    "layer_type": light.get("layer_type"),
                    "layer_category": light.get("layer_category"),
                    "tensor_type": tensor_type,
                    "count": light.get("count"),
                    "min": light.get("min"),
                    "max": light.get("max"),
                    "min_abs_nonzero": light.get("min_abs_nonzero"),
                    "empirical_min_spacing": light.get("empirical_min_spacing"),
                    "frac_zero": light.get("frac_zero"),
                    "format": fmt_res.get("format"),
                    "verdict": fmt_res.get("verdict"),
                    "mean_rel_error": fmt_res.get("mean_rel_error"),
                    "max_rel_error": fmt_res.get("max_rel_error"),
                    "frac_overflow": fmt_res.get("frac_overflow"),
                    "frac_underflow_of_nonzero": fmt_res.get("frac_underflow_of_nonzero"),
                    "scale": fmt_res.get("scale"),
                }
                precision_rows.append(row)

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"stats_summary_{stage_label}.json")
    with open(summary_path, "w") as f:
        json.dump(stats_summary, f, indent=1)

    if precision_rows:
        df = pd.DataFrame(precision_rows)
        csv_path = os.path.join(out_dir, f"precision_table_{stage_label}.csv")
        df.to_csv(csv_path, index=False)

    return stats_summary, precision_rows
