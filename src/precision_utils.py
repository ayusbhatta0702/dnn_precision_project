"""
Simulates casting FP32 data to FP16, FP8 (E4M3 / E5M2), INT16 and INT8
(with per-tensor symmetric scaling) and reports how much would be lost,
purely in numpy so it runs identically whether or not torch happens to
expose native fp8 dtypes.

All functions operate on a 1-D numpy array of FP32 values (typically a
TensorStats reservoir sample) and return a dict of diagnostic numbers, not
just a boolean, so the report can show *why* a verdict was reached.
"""
import numpy as np

# ---- format constants -----------------------------------------------------
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.1035e-5
FP16_MIN_SUBNORMAL = 5.9605e-8

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 2 ** -6          # 0.015625
FP8_E4M3_MIN_SUBNORMAL = 2 ** -9       # smallest subnormal (1 mantissa bit set, 3 mantissa bits)

FP8_E5M2_MAX = 57344.0
FP8_E5M2_MIN_NORMAL = 2 ** -14
FP8_E5M2_MIN_SUBNORMAL = 2 ** -16

INT8_QMAX = 127     # signed, symmetric
INT16_QMAX = 32767   # signed, symmetric

# ---- verdict thresholds (documented, tune as needed) ----------------------
REL_ERR_OK = 0.02          # <=2% mean relative error -> "Sufficient"
REL_ERR_MARGINAL = 0.10    # <=10% -> "Marginal", else "Not sufficient"
OVERFLOW_OK = 0.0          # any overflow to inf is disqualifying for float fmts
UNDERFLOW_OK_FRAC = 0.01   # <=1% of nonzero values flushed to 0 is tolerable


def _rel_err(x, x_hat, eps=1e-12):
    denom = np.maximum(np.abs(x), eps)
    return np.abs(x_hat - x) / denom


# ---------------------------------------------------------------------------
# FP16
# ---------------------------------------------------------------------------
def analyze_fp16(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return _empty_result("fp16")
    with np.errstate(over="ignore", invalid="ignore"):
        x16 = x.astype(np.float16).astype(np.float64)

    overflow = np.isinf(x16) & ~np.isinf(x)
    nz_mask = x != 0
    underflow = nz_mask & (x16 == 0)

    finite_mask = ~overflow
    rel = _rel_err(x[finite_mask], x16[finite_mask]) if finite_mask.any() else np.array([0.0])

    n_overflow = int(overflow.sum())
    n_underflow = int(underflow.sum())
    frac_underflow = n_underflow / max(int(nz_mask.sum()), 1)

    return {
        "format": "FP16",
        "n": int(x.size),
        "n_overflow": n_overflow,
        "frac_overflow": n_overflow / x.size,
        "n_underflow_to_zero": n_underflow,
        "frac_underflow_of_nonzero": frac_underflow,
        "max_rel_error": float(np.max(rel)) if rel.size else 0.0,
        "mean_rel_error": float(np.mean(rel)) if rel.size else 0.0,
        "verdict": _verdict(n_overflow, frac_underflow, float(np.mean(rel)) if rel.size else 0.0),
    }


# ---------------------------------------------------------------------------
# FP8 (E4M3 and E5M2), via native torch cast if available else numpy fallback
# ---------------------------------------------------------------------------
def _fp8_numpy_fallback(x, exp_bits, man_bits, bias, max_val):
    """Round FP64 values to a generic (1, exp_bits, man_bits) float format
    with the given exponent bias, no infinities (finite, saturating), used
    as a fallback if torch's native fp8 dtypes aren't available."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    out = np.zeros_like(ax)

    nz = ax > 0
    if nz.any():
        axnz = ax[nz]
        exp = np.floor(np.log2(axnz))
        min_exp = 1 - bias  # smallest normal exponent
        # subnormal handling: clamp exponent to min_exp, mantissa scaled accordingly
        exp_clamped = np.maximum(exp, min_exp)
        scale = 2.0 ** exp_clamped
        mant = axnz / scale  # in [1,2) for normals, [0,1) region possible for subnormals
        mant_q = np.round(mant * (2 ** man_bits)) / (2 ** man_bits)
        val = mant_q * scale
        val = np.minimum(val, max_val)
        out[nz] = val
    return sign * out


def _fp8_cast(x, variant):
    """Try native torch fp8 dtype cast; fall back to numpy simulation."""
    try:
        import torch
        dtype = torch.float8_e4m3fn if variant == "e4m3" else torch.float8_e5m2
        t = torch.tensor(np.asarray(x, dtype=np.float32))
        t8 = t.to(dtype)
        back = t8.to(torch.float32).numpy().astype(np.float64)
        return back, True
    except Exception:
        if variant == "e4m3":
            back = _fp8_numpy_fallback(x, exp_bits=4, man_bits=3, bias=7, max_val=FP8_E4M3_MAX)
        else:
            back = _fp8_numpy_fallback(x, exp_bits=5, man_bits=2, bias=15, max_val=FP8_E5M2_MAX)
        return back, False


def analyze_fp8(values, variant="e4m3"):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return _empty_result(f"fp8_{variant}")
    max_val = FP8_E4M3_MAX if variant == "e4m3" else FP8_E5M2_MAX
    x8, used_native = _fp8_cast(x, variant)

    nz_mask = x != 0
    overflow = np.abs(x) > max_val
    underflow = nz_mask & (np.abs(x8) == 0)

    rel = _rel_err(x, x8)
    n_overflow = int(overflow.sum())
    n_underflow = int(underflow.sum())
    frac_underflow = n_underflow / max(int(nz_mask.sum()), 1)

    return {
        "format": f"FP8_{variant.upper()}",
        "used_native_torch_cast": used_native,
        "n": int(x.size),
        "n_overflow": n_overflow,
        "frac_overflow": n_overflow / x.size,
        "n_underflow_to_zero": n_underflow,
        "frac_underflow_of_nonzero": frac_underflow,
        "max_rel_error": float(np.max(rel)),
        "mean_rel_error": float(np.mean(rel)),
        "verdict": _verdict(n_overflow, frac_underflow, float(np.mean(rel))),
    }


# ---------------------------------------------------------------------------
# INT16 / INT8 with per-tensor symmetric scaling (standard PTQ approach)
# ---------------------------------------------------------------------------
def analyze_int_quant(values, bits=8):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return _empty_result(f"int{bits}")
    qmax = INT8_QMAX if bits == 8 else INT16_QMAX
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs == 0.0:
        scale = 1.0
    else:
        scale = max_abs / qmax

    q = np.clip(np.round(x / scale), -qmax - 1, qmax)
    deq = q * scale

    abs_err = np.abs(deq - x)
    rel = _rel_err(x, deq)

    # dynamic-range utilization: how many of the 2^bits codes are "wasted"
    # because the smallest nonzero magnitude needs a much finer scale than
    # the largest magnitude does (classic PTQ range-vs-precision tradeoff)
    nz = x[x != 0]
    min_abs_nonzero = float(np.min(np.abs(nz))) if nz.size else None
    dynamic_range_ratio = (max_abs / min_abs_nonzero) if min_abs_nonzero else None
    representable_levels = 2 * qmax + 1

    return {
        "format": f"INT{bits}",
        "n": int(x.size),
        "scale": scale,
        "qmax_code": qmax,
        "max_abs_value": max_abs,
        "min_abs_nonzero_value": min_abs_nonzero,
        "dynamic_range_ratio": dynamic_range_ratio,
        "representable_levels": representable_levels,
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "max_rel_error": float(np.max(rel)),
        "mean_rel_error": float(np.mean(rel)),
        "verdict": _verdict(0, 0.0, float(np.mean(rel))),
    }


def _verdict(n_overflow, frac_underflow, mean_rel_error):
    if n_overflow > 0 or frac_underflow > UNDERFLOW_OK_FRAC:
        return "Not sufficient"
    if mean_rel_error <= REL_ERR_OK:
        return "Sufficient"
    if mean_rel_error <= REL_ERR_MARGINAL:
        return "Marginal"
    return "Not sufficient"


def _empty_result(fmt):
    return {"format": fmt, "n": 0, "verdict": "N/A (no data)"}


def full_precision_analysis(values):
    """Run all four candidate formats (FP8 in both variants) on one sample
    and return a combined dict, ready to be written to the precision csv."""
    values = np.asarray(values)
    return {
        "fp16": analyze_fp16(values),
        "fp8_e4m3": analyze_fp8(values, "e4m3"),
        "fp8_e5m2": analyze_fp8(values, "e5m2"),
        "int16": analyze_int_quant(values, bits=16),
        "int8": analyze_int_quant(values, bits=8),
    }
