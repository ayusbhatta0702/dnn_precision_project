# DNN Numerical Range & Precision Analysis

End-to-end project to study the numerical range/precision of tensors flowing
through a DNN during **training** (ResNet-18 / CIFAR-10) and **inference**
(ResNet-18/CIFAR-10, MobileNetV2/CIFAR-100, LeNet-5/MNIST), and to judge
whether FP16, FP8, INT16 or INT8 could replace FP32 for each tensor type.

Everything runs in FP32. No other dtype is ever used for compute — FP16 /
FP8 / INT16 / INT8 are only *simulated in numpy* on the collected FP32 data
to measure how much would be lost (overflow / underflow / quantization
error) if you switched to them.

---
## 1. Project layout

```
dnn_precision_project/
├── requirements.txt
├── run_all.sh                     convenience script: runs everything
└── src/
    ├── models.py                  ResNet-18 (CIFAR stem), MobileNetV2 (CIFAR stem), LeNet-5
    ├── data.py                    CIFAR-10 / CIFAR-100 / MNIST loaders (torchvision)
    ├── stats_collector.py         streaming statistics + reservoir sampling per tensor
    ├── hooks.py                   forward/backward hooks that feed stats_collector
    ├── precision_utils.py         FP16 / FP8(e4m3,e5m2) / INT16 / INT8 sufficiency checks
    ├── plotting.py                full-range + zoomed-near-zero distribution plots
    ├── train_and_collect.py       TRAINING analysis (ResNet-18 / CIFAR-10)
    ├── inference_and_collect.py   INFERENCE analysis (3 model/dataset combos)
    └── analyze_report.py          builds the final consolidated Markdown report
```

All results (raw stats JSON, PNG plots, CSV tables, final report) are written
under `outputs/<run_name>/`.

---
## 2. Setup

Tested with Python 3.10+.

```bash
cd dnn_precision_project
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs CPU PyTorch by default. If you have a GPU, install
the matching CUDA build of torch/torchvision from https://pytorch.org first,
then `pip install -r requirements.txt` for the rest (numpy/matplotlib/pandas
will just be skipped if already satisfied).

Datasets (CIFAR-10, CIFAR-100, MNIST) are downloaded automatically by
torchvision on first run (~200 MB total) — an internet connection is needed
the first time only.

---
## 3. Running

### One command, everything
```bash
bash run_all.sh
```
This runs the training analysis, the three inference analyses, and finally
builds the consolidated report at `outputs/FINAL_REPORT.md`.
On a laptop CPU expect ~20–40 minutes total with the default light settings
below; a GPU brings this down to a few minutes.

### Or run each stage yourself

**a) Training analysis (ResNet-18 / CIFAR-10)**
```bash
python3 src/train_and_collect.py \
    --epochs 8 --batch-size 128 --lr 0.05 \
    --data-root ./data --out outputs/train_resnet18_cifar10
```
- Trains ResNet-18 on CIFAR-10 for `--epochs` epochs (default 8 — enough for
  the network to move well away from its random-init regime; raise this for
  a more fully-converged "end of training" snapshot).
- At three points — **beginning** (before the first optimizer step), **middle**
  (epoch ≈ epochs/2) and **end** (last epoch) — it runs a handful of batches
  with hooks attached and records, *per layer*: layer inputs, weights,
  activations (layer outputs), weight gradients, weight updates
  (Δw = w_after − w_before one optimizer step), and biases.
- Writes per-layer/per-stage/per-tensor-type: summary stats (json), a full
  distribution plot and a zoomed-near-zero plot (png), and a precision
  sufficiency verdict (csv).

**b) Inference analysis (3 model/dataset pairs)**
```bash
python3 src/inference_and_collect.py --pair resnet18_cifar10   --epochs 5 --out outputs/infer_resnet18_cifar10
python3 src/inference_and_collect.py --pair mobilenetv2_cifar100 --epochs 5 --out outputs/infer_mobilenetv2_cifar100
python3 src/inference_and_collect.py --pair lenet5_mnist       --epochs 5 --out outputs/infer_lenet5_mnist
```
- `--epochs` briefly trains the model first so its weights/activations are
  representative of a working network rather than pure random init (LeNet-5/
  MNIST reaches ~98–99% in 5 epochs; the CIFAR models will not be fully
  converged in 5 epochs — raise `--epochs` if you want production-accuracy
  weights; the range/precision conclusions are not very sensitive to this).
- Then performs **one full pass over the entire test set** with forward
  hooks attached, streaming-accumulating statistics for inputs, weights
  (collected once), activations, biases (collected once), and output
  logits — for every Conv2d / Linear / BatchNorm2d / activation / pooling
  layer.
- Same outputs as above: json stats, full + zoomed plots, precision csv.

**c) Consolidated report**
```bash
python3 src/analyze_report.py --out outputs
```
Reads every `outputs/*/summary.json` and `outputs/*/precision_table.csv`
produced above and writes `outputs/FINAL_REPORT.md`: one table per tensor
type (input/weight/activation/gradient/weight-update/bias/logits) showing
min, max, smallest nonzero magnitude, empirical spacing between distinct
values, and the FP16/FP8/INT16/INT8 verdicts, plus a written discussion.

---
## 4. What "sufficiency" means here (`precision_utils.py`)

For every collected FP32 tensor (or its reservoir sample, for huge
activation streams) we simulate each candidate format on the *exact same
values* and report:

- **FP16 (1-5-10)**: max representable magnitude ≈ 6.55×10⁴, smallest
  normal ≈ 6.1×10⁻⁵, smallest subnormal ≈ 5.96×10⁻⁸. We report the count/
  fraction of values that would overflow to ±inf, underflow to 0, and the
  max/mean relative round-trip error.
- **FP8**: both common variants are checked —
  **E4M3** (max ≈ 448, smallest normal ≈ 2⁻⁶) and
  **E5M2** (max ≈ 57344, smallest normal ≈ 2⁻¹⁴, coarser mantissa).
  If your installed PyTorch exposes `torch.float8_e4m3fn` / `torch.float8_e5m2`
  the real hardware-accurate cast is used; otherwise a numpy fallback that
  implements the same exponent/mantissa bit budget is used.
- **INT16 / INT8 (with scaling)**: values are affine/symmetric-quantized as
  DNN quantization toolchains do: `scale = max(|x|) / (2^(bits-1) - 1)`,
  `x_q = round(x / scale)` clipped to the signed integer range, then
  dequantized. We report the chosen scale, max/mean absolute and relative
  dequantization error, and how many effective bits of the dynamic range are
  "wasted" (ratio of max to smallest-nonzero magnitude vs. 2^bits).
- A **verdict** ("Sufficient" / "Marginal" / "Not sufficient") is assigned
  per tensor type using fixed thresholds on relative error and
  overflow/underflow fraction (documented at the top of
  `precision_utils.py`) — these are only a starting point; the numeric
  columns are what actually matter, and you're expected to eyeball the plots
  too.

---
## 5. Plots

For every (layer × tensor-type × stage) we save two PNGs:
- `..._full.png` — histogram (density-normalized) over the entire observed
  range, x-axis = value, y-axis = probability density.
- `..._zoom.png` — the same data restricted to a window around zero
  (±2% of the largest magnitude observed, or tighter if the data warrants
  it) so closely-spaced small values are visible.

Both plots' titles include min, max, smallest-nonzero-magnitude and the
empirical minimum spacing between distinct sampled values.

---
## 6. Notes on the streaming/sampling approach

The assignment requires activations/logits to be collected **across the
entire test set**. Storing every activation value for every layer over
10,000 test images (a single ResNet-18 conv layer alone produces tens of
millions of values per image) is not memory-feasible, so `stats_collector.py`
uses **exact streaming statistics** (running min, max, count, zero-count,
smallest-nonzero-magnitude — computed over *every* value, not a sample) plus
a **bounded reservoir sample** (default 2,000,000 values per layer/tensor
type, sampled uniformly across all batches) that is used only for the
distribution plots and the empirical "spacing between distinct values"
figure. This keeps memory bounded while keeping min/max/overflow/underflow
figures exact. You can raise `--reservoir-size` if you want smoother plots
at the cost of more RAM.

---
## 7. Extending / adapting

- Swap in a different model of similar complexity by adding it to
  `models.py` and registering it in `PAIRS` in `inference_and_collect.py`.
- `hooks.py`'s `TARGET_LAYER_TYPES` controls which layer types are
  instrumented (Conv2d, Linear, BatchNorm2d, ReLU/activations, pooling) —
  add/remove types there.
