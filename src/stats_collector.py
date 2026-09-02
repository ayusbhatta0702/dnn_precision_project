"""
TensorStats: memory-bounded streaming statistics for a (potentially huge)
sequence of tensor values, fed in batch-by-batch as numpy arrays.

Tracks EXACTLY (over every value ever seen, no sampling):
    - count, count_zero, count_negative, count_positive
    - global min, max
    - smallest nonzero absolute magnitude
largest-magnitude and smallest-magnitude values are exact; the *shape* of
the distribution (for plotting) and the empirical "spacing between distinct
values" figure are estimated from a bounded reservoir sample (uniform
sample across the whole stream), since storing every value for an entire
test-set pass is not memory feasible for large activation tensors.
"""
import numpy as np


class TensorStats:
    def __init__(self, name, reservoir_size=2_000_000, seed=0):
        self.name = name
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)

        self.count = 0
        self.count_zero = 0
        self.count_negative = 0
        self.count_positive = 0
        self.min_val = np.inf
        self.max_val = -np.inf
        self.min_abs_nonzero = np.inf
        self.sum_ = 0.0
        self.sum_sq = 0.0

        # reservoir: fixed-capacity buffer, filled via weighted random
        # subsampling so that every value seen across the whole stream has
        # (approximately) equal probability of ending up in the buffer.
        self._reservoir = np.empty(reservoir_size, dtype=np.float64)
        self._reservoir_fill = 0
        self._n_seen_for_reservoir = 0  # total elements offered so far

    def update(self, arr):
        """arr: numpy array (any shape/dtype), float values. NaN/Inf are
        recorded in count but excluded from min/max/reservoir (flagged
        separately)."""
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            return
        arr = arr.astype(np.float64, copy=False)

        finite_mask = np.isfinite(arr)
        n_nonfinite = arr.size - int(finite_mask.sum())
        if n_nonfinite:
            if not hasattr(self, "count_nonfinite"):
                self.count_nonfinite = 0
            self.count_nonfinite += n_nonfinite
        arr = arr[finite_mask]
        if arr.size == 0:
            self.count += int(finite_mask.size)
            return

        self.count += arr.size
        self.count_zero += int((arr == 0).sum())
        self.count_negative += int((arr < 0).sum())
        self.count_positive += int((arr > 0).sum())

        batch_min = float(arr.min())
        batch_max = float(arr.max())
        self.min_val = min(self.min_val, batch_min)
        self.max_val = max(self.max_val, batch_max)

        nz = arr[arr != 0]
        if nz.size:
            batch_min_abs = float(np.abs(nz).min())
            self.min_abs_nonzero = min(self.min_abs_nonzero, batch_min_abs)

        self.sum_ += float(arr.sum())
        self.sum_sq += float(np.square(arr).sum())

        self._reservoir_update(arr)

    def _reservoir_update(self, arr):
        """Distributed reservoir sampling: every offered element has equal
        probability reservoir_size / n_seen_for_reservoir of being kept,
        implemented per-batch with vectorized random selection (a close,
        practical approximation to classic Algorithm R, adequate for
        building representative histograms)."""
        n = arr.size
        cap = self.reservoir_size
        if self._reservoir_fill < cap:
            take = min(n, cap - self._reservoir_fill)
            # take a random subset of this batch (not just the prefix) so
            # within-batch order doesn't bias the fill phase
            idx = self.rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
            self._reservoir[self._reservoir_fill:self._reservoir_fill + take] = arr[idx]
            self._reservoir_fill += take
            remaining = n - take
            arr = np.delete(arr, idx) if remaining > 0 else arr[:0]
            n = arr.size

        self._n_seen_for_reservoir += n
        if n == 0:
            return
        # once full: keep a running uniform sample by randomly swapping in
        # a proportionally-sized random subset of the new batch
        keep_prob = cap / max(self._n_seen_for_reservoir, cap)
        n_swap = int(self.rng.binomial(min(n, cap), keep_prob)) if n > 0 else 0
        if n_swap > 0:
            new_vals_idx = self.rng.choice(n, size=n_swap, replace=False)
            slot_idx = self.rng.choice(cap, size=n_swap, replace=False)
            self._reservoir[slot_idx] = arr[new_vals_idx]

    @property
    def sample(self):
        return self._reservoir[:self._reservoir_fill]

    def summary(self):
        s = self.sample
        mean = self.sum_ / self.count if self.count else float("nan")
        var = (self.sum_sq / self.count - mean ** 2) if self.count else float("nan")
        std = float(np.sqrt(max(var, 0.0)))

        spacing = None
        if s.size >= 2:
            uniq = np.unique(s)
            if uniq.size >= 2:
                diffs = np.diff(uniq)
                diffs = diffs[diffs > 0]
                if diffs.size:
                    spacing = float(np.min(diffs))

        return {
            "name": self.name,
            "count": int(self.count),
            "count_zero": int(self.count_zero),
            "frac_zero": (self.count_zero / self.count) if self.count else None,
            "count_nonfinite": int(getattr(self, "count_nonfinite", 0)),
            "min": float(self.min_val) if np.isfinite(self.min_val) else None,
            "max": float(self.max_val) if np.isfinite(self.max_val) else None,
            "min_abs_nonzero": float(self.min_abs_nonzero) if np.isfinite(self.min_abs_nonzero) else None,
            "mean": mean,
            "std": std,
            "empirical_min_spacing": spacing,
            "reservoir_size_used": int(s.size),
        }

    def to_dict_with_sample(self, max_sample_out=200_000):
        """Summary + a (further subsampled if needed) copy of the reservoir,
        for saving to disk / plotting."""
        d = self.summary()
        s = self.sample
        if s.size > max_sample_out:
            idx = self.rng.choice(s.size, size=max_sample_out, replace=False)
            s = s[idx]
        d["sample"] = s.tolist()
        return d
