"""
TRAINING analysis: trains ResNet-18 on CIFAR-10 in plain FP32 and, at three
points during training (beginning / middle / end), attaches hooks for a
handful of batches to record, per layer:
    Inputs, Weights, Activations, Gradients (weight grad), Weight updates
    (delta-w over one optimizer step), Bias values

Usage:
    python3 src/train_and_collect.py --epochs 8 --out outputs/train_resnet18_cifar10
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models import resnet18_cifar
from data import get_cifar10_loaders
from hooks import StatsRegistry
from collect_utils import process_registry


def run_stage(model, loader_iter, optimizer, criterion, device, registry,
              n_batches, collect_weights_biases=True):
    """Runs n_batches of real train steps with hooks attached, updating
    `registry` with input/activation/weight-gradient/weight-update stats.
    Weights & biases are snapshotted once, at the start of the stage."""
    registry.register_forward(model)
    if collect_weights_biases:
        registry.collect_weights_and_biases(model)

    model.train()
    for i in range(n_batches):
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            break
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)          # forward hooks fire here -> input/activation stats
        loss = criterion(outputs, labels)
        loss.backward()
        registry.collect_weight_gradients(model)

        pre_snapshot = registry.snapshot_weights(model)
        optimizer.step()
        registry.collect_weight_updates(model, pre_snapshot)

    registry.remove_forward_hooks()
    return registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--stage-batches", type=int, default=5,
                     help="number of real train batches used to collect stats at each stage")
    ap.add_argument("--reservoir-size", type=int, default=1_000_000)
    ap.add_argument("--out", default="outputs/train_resnet18_cifar10")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_and_collect] device={device}")
    if args.epochs < 3:
        print("[train_and_collect] WARNING: --epochs < 3 means the begin/middle/end "
              "stages may coincide in the same epoch; 3+ epochs recommended.")

    os.makedirs(args.out, exist_ok=True)
    train_loader, test_loader = get_cifar10_loaders(args.data_root, batch_size=args.batch_size)

    model = resnet18_cifar(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    mid_epoch = max(args.epochs // 2, 1)
    last_epoch = args.epochs

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        is_begin_epoch = (epoch == 1)
        is_mid_epoch = (epoch == mid_epoch)
        is_end_epoch = (epoch == last_epoch)

        loader_iter = iter(train_loader)
        n_batches_this_epoch = len(train_loader)

        if is_begin_epoch:
            print(f"[epoch {epoch}] collecting STAGE=begin ({args.stage_batches} batches)")
            reg = StatsRegistry(reservoir_size=args.reservoir_size)
            run_stage(model, loader_iter, optimizer, criterion, device, reg, args.stage_batches)
            process_registry(reg, args.out, stage_label="begin")
            n_batches_this_epoch -= args.stage_batches

        if is_mid_epoch:
            print(f"[epoch {epoch}] collecting STAGE=middle ({args.stage_batches} batches)")
            reg = StatsRegistry(reservoir_size=args.reservoir_size)
            run_stage(model, loader_iter, optimizer, criterion, device, reg, args.stage_batches)
            process_registry(reg, args.out, stage_label="middle")
            n_batches_this_epoch -= args.stage_batches

        # run the remaining ordinary batches of this epoch without hooks (fast)
        model.train()
        for i in range(n_batches_this_epoch):
            try:
                images, labels = next(loader_iter)
            except StopIteration:
                break

            # for the final epoch, collect STAGE=end stats on the very last
            # `stage_batches` batches instead of running them "plain"
            remaining_including_this = n_batches_this_epoch - i
            if is_end_epoch and remaining_including_this == args.stage_batches:
                print(f"[epoch {epoch}] collecting STAGE=end ({args.stage_batches} batches)")
                reg = StatsRegistry(reservoir_size=args.reservoir_size)
                remaining_iter = _prepend(images, labels, loader_iter)
                run_stage(model, remaining_iter, optimizer, criterion, device, reg,
                          args.stage_batches, collect_weights_biases=True)
                process_registry(reg, args.out, stage_label="end")
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # quick test accuracy for sanity/logging
        acc = evaluate(model, test_loader, device)
        print(f"[epoch {epoch}/{args.epochs}] test_acc={acc:.4f} "
              f"elapsed={time.time()-t0:.0f}s")

    torch.save(model.state_dict(), os.path.join(args.out, "final_model.pt"))
    print(f"[train_and_collect] done. outputs in {args.out}")


def _prepend(images, labels, it):
    """Yield the already-fetched (images, labels) batch first, then
    continue draining the given iterator -- so STAGE=end can reuse the
    batch that was fetched to detect 'we're near the end of the epoch'."""
    yield images, labels
    for batch in it:
        yield batch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    main()
