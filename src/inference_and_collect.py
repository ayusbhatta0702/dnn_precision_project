"""
INFERENCE analysis: for one of three model/dataset pairs, briefly trains the
model (so weights/activations are representative of a working network, not
random init), then performs ONE COMPLETE PASS over the entire test set in
FP32, with hooks attached, collecting per layer: layer inputs, weights
(once), activations, biases (once), and the network's output logits.

Usage:
    python3 src/inference_and_collect.py --pair resnet18_cifar10 --epochs 5 \
        --out outputs/infer_resnet18_cifar10
    python3 src/inference_and_collect.py --pair mobilenetv2_cifar100 --epochs 5 \
        --out outputs/infer_mobilenetv2_cifar100
    python3 src/inference_and_collect.py --pair lenet5_mnist --epochs 5 \
        --out outputs/infer_lenet5_mnist
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models import resnet18_cifar, mobilenetv2_cifar, lenet5
from data import get_cifar10_loaders, get_cifar100_loaders, get_mnist_loaders
from hooks import StatsRegistry
from collect_utils import process_registry

PAIRS = {
    "resnet18_cifar10": {
        "model": lambda: resnet18_cifar(num_classes=10),
        "loaders": get_cifar10_loaders,
    },
    "mobilenetv2_cifar100": {
        "model": lambda: mobilenetv2_cifar(num_classes=100),
        "loaders": get_cifar100_loaders,
    },
    "lenet5_mnist": {
        "model": lambda: lenet5(num_classes=10),
        "loaders": get_mnist_loaders,
    },
}


def quick_train(model, train_loader, device, epochs, lr=0.05):
    if epochs <= 0:
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        running_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        scheduler.step()
        print(f"  [pretrain epoch {epoch}/{epochs}] loss={running_loss/max(n,1):.4f} "
              f"elapsed={time.time()-t0:.0f}s")


@torch.no_grad()
def full_test_pass(model, test_loader, device, registry):
    registry.register_forward(model)
    registry.collect_weights_and_biases(model)

    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)              # forward hooks fire -> input/activation stats
        registry.update("output", "logit", outputs)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    registry.remove_forward_hooks()
    acc = correct / max(total, 1)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=5, help="quick pretraining epochs before inference")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reservoir-size", type=int, default=2_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = args.out or f"outputs/infer_{args.pair}"
    os.makedirs(out_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference_and_collect] pair={args.pair} device={device}")

    spec = PAIRS[args.pair]
    train_loader, test_loader = spec["loaders"](args.data_root, batch_size=args.batch_size)
    model = spec["model"]().to(device)

    print(f"[inference_and_collect] quick-training for {args.epochs} epoch(s)...")
    quick_train(model, train_loader, device, args.epochs, lr=args.lr)

    print("[inference_and_collect] running full test-set pass with hooks attached...")
    registry = StatsRegistry(reservoir_size=args.reservoir_size)
    acc = full_test_pass(model, test_loader, device, registry)
    print(f"[inference_and_collect] test accuracy = {acc:.4f}")

    print("[inference_and_collect] computing stats / plots / precision tables...")
    process_registry(registry, out_dir, stage_label="inference")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "test_accuracy.txt"), "w") as f:
        f.write(f"{acc:.4f}\n")
    print(f"[inference_and_collect] done. outputs in {out_dir}")


if __name__ == "__main__":
    main()
