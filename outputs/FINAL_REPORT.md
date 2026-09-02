# DNN FP32 Range/Precision Analysis — Consolidated Report

Runs included: ['infer_lenet5_mnist', 'infer_mobilenetv2_cifar100', 'infer_resnet18_cifar10', 'train_resnet18_cifar10']


## 1. Range / precision summary by tensor type

Aggregated across every layer and run that collected this tensor type. `smallest_abs_nonzero` and `median_empirical_spacing` are computed from the reservoir samples (see README, section 6).


### input

- layers/tensors covered: **208**

- global min: **-168.157**   global max: **187.124**

- smallest |nonzero| observed: **5.95876e-13**

- median empirical spacing between distinct sampled values: **4.65661e-10**

- largest fraction-of-zeros seen in any single layer: **0.7750**


### weight

- layers/tensors covered: **150**

- global min: **-0.963461**   global max: **1.62793**

- smallest |nonzero| observed: **1.43372e-11**

- median empirical spacing between distinct sampled values: **4.65661e-08**

- largest fraction-of-zeros seen in any single layer: **0.0000**


### activation

- layers/tensors covered: **208**

- global min: **-168.157**   global max: **187.124**

- smallest |nonzero| observed: **5.95876e-13**

- median empirical spacing between distinct sampled values: **4.65661e-10**

- largest fraction-of-zeros seen in any single layer: **0.7750**


### weight_gradient

- layers/tensors covered: **41**

- global min: **-0.184468**   global max: **0.36881**

- smallest |nonzero| observed: **9.09495e-13**

- median empirical spacing between distinct sampled values: **5.82077e-11**

- largest fraction-of-zeros seen in any single layer: **0.0000**


### weight_update

- layers/tensors covered: **41**

- global min: **-0.0132087**   global max: **0.00943781**

- smallest |nonzero| observed: **5.45697e-12**

- median empirical spacing between distinct sampled values: **4.65661e-10**

- largest fraction-of-zeros seen in any single layer: **0.0063**


### bias

- layers/tensors covered: **79**

- global min: **-0.432828**   global max: **0.98632**

- smallest |nonzero| observed: **6.72475e-10**

- median empirical spacing between distinct sampled values: **3.29688e-07**

- largest fraction-of-zeros seen in any single layer: **1.0000**


### logit

- layers/tensors covered: **1**

- global min: **-14.262**   global max: **26.4843**

- smallest |nonzero| observed: **4.448e-06**

- median empirical spacing between distinct sampled values: **7.45058e-09**

- largest fraction-of-zeros seen in any single layer: **0.0000**


## 2. FP16 / FP8 / INT16 / INT8 verdict counts by tensor type

Each cell = number of (layer × run) instances of this tensor type that received that verdict for that format. See `precision_utils.py` for the exact thresholds.


### input

| format   |   Marginal |   Not sufficient |   Sufficient |
|:---------|-----------:|-----------------:|-------------:|
| FP16     |          0 |                0 |          388 |
| FP8_E4M3 |        144 |               16 |          228 |
| FP8_E5M2 |        373 |                0 |           15 |
| INT16    |          0 |                0 |          388 |
| INT8     |        332 |               49 |            7 |


### weight

| format   |   Marginal |   Not sufficient |   Sufficient |
|:---------|-----------:|-----------------:|-------------:|
| FP16     |          0 |                0 |          274 |
| FP8_E4M3 |         86 |              102 |           86 |
| FP8_E5M2 |        209 |                0 |           65 |
| INT16    |          0 |                0 |          274 |
| INT8     |        121 |                0 |          153 |


### activation

| format   |   Marginal |   Not sufficient |   Sufficient |
|:---------|-----------:|-----------------:|-------------:|
| FP16     |          0 |                0 |          388 |
| FP8_E4M3 |        263 |               16 |          109 |
| FP8_E5M2 |        380 |                0 |            8 |
| INT16    |          0 |                0 |          388 |
| INT8     |        314 |               73 |            1 |


### weight_gradient

| format   |   Marginal |   Not sufficient |   Sufficient |
|:---------|-----------:|-----------------:|-------------:|
| FP16     |          0 |                0 |          123 |
| FP8_E4M3 |          1 |              122 |            0 |
| FP8_E5M2 |        107 |               16 |            0 |
| INT16    |          0 |                0 |          123 |
| INT8     |        109 |               14 |            0 |


### weight_update

| format   |   Marginal |   Not sufficient |   Sufficient |
|:---------|-----------:|-----------------:|-------------:|
| FP16     |          5 |                4 |          114 |
| FP8_E4M3 |          0 |              123 |            0 |
| FP8_E5M2 |          9 |              114 |            0 |
| INT16    |          0 |                0 |          123 |
| INT8     |        117 |                1 |            5 |


### bias

| format   |   Marginal |   Not sufficient |   Sufficient |
|:---------|-----------:|-----------------:|-------------:|
| FP16     |          4 |                2 |          136 |
| FP8_E4M3 |         36 |               85 |           21 |
| FP8_E5M2 |        105 |               17 |           20 |
| INT16    |          0 |                0 |          142 |
| INT8     |         78 |                3 |           61 |


### logit

| format   |   Marginal |   Sufficient |
|:---------|-----------:|-------------:|
| FP16     |          0 |            3 |
| FP8_E4M3 |          3 |            0 |
| FP8_E5M2 |          3 |            0 |
| INT16    |          0 |            3 |
| INT8     |          3 |            0 |


## 3. Per-run detail


### infer_lenet5_mnist

stages/tensor types collected: ['inference'] / ['activation', 'bias', 'input', 'logit', 'weight']

Full per-layer numbers: `outputs/infer_lenet5_mnist/precision_table_*.csv`  

Plots: `outputs/infer_lenet5_mnist/plots/<stage>/<layer>/<tensor_type>_{full,zoom}.png`


### infer_mobilenetv2_cifar100

stages/tensor types collected: ['inference'] / ['activation', 'bias', 'input', 'logit', 'weight']

Full per-layer numbers: `outputs/infer_mobilenetv2_cifar100/precision_table_*.csv`  

Plots: `outputs/infer_mobilenetv2_cifar100/plots/<stage>/<layer>/<tensor_type>_{full,zoom}.png`


### infer_resnet18_cifar10

stages/tensor types collected: ['inference'] / ['activation', 'bias', 'input', 'logit', 'weight']

Full per-layer numbers: `outputs/infer_resnet18_cifar10/precision_table_*.csv`  

Plots: `outputs/infer_resnet18_cifar10/plots/<stage>/<layer>/<tensor_type>_{full,zoom}.png`


### train_resnet18_cifar10

stages/tensor types collected: ['begin', 'end', 'middle'] / ['activation', 'bias', 'input', 'weight', 'weight_gradient', 'weight_update']

Full per-layer numbers: `outputs/train_resnet18_cifar10/precision_table_*.csv`  

Plots: `outputs/train_resnet18_cifar10/plots/<stage>/<layer>/<tensor_type>_{full,zoom}.png`


## 4. Discussion (fill in after reviewing the numbers/plots above)


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
