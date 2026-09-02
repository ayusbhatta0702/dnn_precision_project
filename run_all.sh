#!/usr/bin/env bash
# Runs the full training + inference + report pipeline with sensible
# defaults. Edit the flags below (epochs especially) to trade off runtime
# vs. how well-converged the models are.
set -e
cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:-./data}
OUT=${OUT:-outputs}

echo "=== [1/5] Training analysis: ResNet-18 / CIFAR-10 ==="
python3 src/train_and_collect.py \
    --data-root "$DATA_ROOT" --epochs 8 --batch-size 128 --lr 0.05 \
    --out "$OUT/train_resnet18_cifar10"

echo "=== [2/5] Inference analysis: ResNet-18 / CIFAR-10 ==="
python3 src/inference_and_collect.py --pair resnet18_cifar10 \
    --data-root "$DATA_ROOT" --epochs 5 --out "$OUT/infer_resnet18_cifar10"

echo "=== [3/5] Inference analysis: MobileNetV2 / CIFAR-100 ==="
python3 src/inference_and_collect.py --pair mobilenetv2_cifar100 \
    --data-root "$DATA_ROOT" --epochs 5 --out "$OUT/infer_mobilenetv2_cifar100"

echo "=== [4/5] Inference analysis: LeNet-5 / MNIST ==="
python3 src/inference_and_collect.py --pair lenet5_mnist \
    --data-root "$DATA_ROOT" --epochs 5 --out "$OUT/infer_lenet5_mnist"

echo "=== [5/5] Building consolidated report ==="
python3 src/analyze_report.py --out "$OUT"

echo
echo "Done. See $OUT/FINAL_REPORT.md and the plots/ subfolders under each $OUT/<run>/."
