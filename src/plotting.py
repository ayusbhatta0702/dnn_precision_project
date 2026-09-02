"""
Distribution plots for a collected sample of values:
  - "full": histogram over the entire observed range, x = value, y = density
  - "zoom": same data restricted to a window around zero, to see closely
            spaced small values

Both are saved as PNG. Matplotlib only (headless/Agg backend).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _zoom_window(sample, min_val, max_val):
    """Pick a symmetric window around 0 that reveals fine structure near
    zero: 2% of the largest observed magnitude, but never wider than the
    full range and never narrower than a few multiples of the smallest
    nonzero magnitude in-sample."""
    max_abs = max(abs(min_val), abs(max_val), 1e-30)
    r = 0.02 * max_abs
    nz = sample[sample != 0]
    if nz.size:
        smallest = np.percentile(np.abs(nz), 1)
        r = max(r, smallest * 20)
    r = min(r, max_abs)
    if r <= 0:
        r = max_abs if max_abs > 0 else 1.0
    return -r, r


def plot_distribution(sample, title, out_prefix, stats=None, bins=200):
    """sample: 1D numpy array of values (e.g. a TensorStats reservoir sample).
    out_prefix: path prefix; writes f"{out_prefix}_full.png" and
    f"{out_prefix}_zoom.png". Returns dict with the two file paths."""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    sample = np.asarray(sample)
    sample = sample[np.isfinite(sample)]

    result = {"full": None, "zoom": None}
    if sample.size == 0:
        return result

    min_val, max_val = float(sample.min()), float(sample.max())
    nz = sample[sample != 0]
    min_abs_nz = float(np.abs(nz).min()) if nz.size else None

    subtitle = f"min={min_val:.4g}  max={max_val:.4g}"
    if min_abs_nz is not None:
        subtitle += f"  min|nz|={min_abs_nz:.4g}"
    if stats and stats.get("empirical_min_spacing") is not None:
        subtitle += f"  min spacing~{stats['empirical_min_spacing']:.4g}"

    # ---- full range ----
    fig, ax = plt.subplots(figsize=(7, 4))
    if max_val > min_val:
        ax.hist(sample, bins=bins, density=True, color="#3366cc", alpha=0.85)
    else:
        ax.axvline(min_val, color="#3366cc")
    ax.set_xlabel("value")
    ax.set_ylabel("probability density")
    ax.set_title(f"{title} — full range\n{subtitle}", fontsize=9)
    fig.tight_layout()
    full_path = f"{out_prefix}_full.png"
    fig.savefig(full_path, dpi=130)
    plt.close(fig)
    result["full"] = full_path

    # ---- zoomed near zero ----
    lo, hi = _zoom_window(sample, min_val, max_val)
    zmask = (sample >= lo) & (sample <= hi)
    zsample = sample[zmask]
    fig, ax = plt.subplots(figsize=(7, 4))
    if zsample.size >= 2 and np.ptp(zsample) > 0:
        ax.hist(zsample, bins=bins, density=True, color="#cc3333", alpha=0.85)
    elif zsample.size:
        ax.axvline(float(zsample[0]), color="#cc3333")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("value")
    ax.set_ylabel("probability density")
    frac_in_window = zsample.size / sample.size
    ax.set_title(f"{title} — zoom near zero [{lo:.3g}, {hi:.3g}]\n"
                 f"{frac_in_window*100:.1f}% of samples in window   {subtitle}", fontsize=9)
    fig.tight_layout()
    zoom_path = f"{out_prefix}_zoom.png"
    fig.savefig(zoom_path, dpi=130)
    plt.close(fig)
    result["zoom"] = zoom_path

    return result
