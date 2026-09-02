"""
Registers forward hooks on the layer types we care about (Conv2d, Linear,
BatchNorm2d, activation layers, pooling layers) and routes their
input/output/weight/bias tensors into TensorStats collectors, keyed by
(layer_name, tensor_type). Also provides helpers for weight-gradient and
weight-update collection around an optimizer step.

Only layer *types* are targeted, not every nn.Module, so container modules
(Sequential, the ResNet BasicBlock wrapper, etc.) are skipped automatically.
"""
import torch
import torch.nn as nn

from stats_collector import TensorStats

TARGET_LAYER_TYPES = (
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.ReLU6,
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
)

# human-readable category for each torch layer type, used just for labeling
_LAYER_CATEGORY = {
    nn.Conv2d: "conv",
    nn.Linear: "linear",
    nn.BatchNorm2d: "batchnorm",
    nn.ReLU: "activation",
    nn.ReLU6: "activation",
    nn.AvgPool2d: "pooling",
    nn.MaxPool2d: "pooling",
    nn.AdaptiveAvgPool2d: "pooling",
}


def layer_category(module):
    for t, cat in _LAYER_CATEGORY.items():
        if isinstance(module, t):
            return cat
    return "other"


class StatsRegistry:
    """Owns one TensorStats per (layer_name, tensor_type) key, created
    lazily on first use, and the set of registered hook handles."""

    def __init__(self, reservoir_size=2_000_000, seed=0):
        self.reservoir_size = reservoir_size
        self.seed = seed
        self.stats = {}          # (layer_name, tensor_type) -> TensorStats
        self.layer_meta = {}     # layer_name -> {"type": str, "category": str}
        self._handles = []

    def get(self, layer_name, tensor_type):
        key = (layer_name, tensor_type)
        if key not in self.stats:
            self.stats[key] = TensorStats(f"{layer_name}.{tensor_type}",
                                           reservoir_size=self.reservoir_size,
                                           seed=self.seed)
        return self.stats[key]

    def update(self, layer_name, tensor_type, tensor):
        if tensor is None:
            return
        arr = tensor.detach().to("cpu", torch.float32).numpy()
        self.get(layer_name, tensor_type).update(arr)

    # ---- forward hooks: layer input, activation (=layer output) ----------
    def register_forward(self, model, include_input=True, include_activation=True):
        for name, module in model.named_modules():
            if not isinstance(module, TARGET_LAYER_TYPES):
                continue
            self.layer_meta[name] = {
                "type": type(module).__name__,
                "category": layer_category(module),
            }

            def make_hook(layer_name):
                def hook(mod, inputs, output):
                    if include_input and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                        self.update(layer_name, "input", inputs[0])
                    if include_activation and isinstance(output, torch.Tensor):
                        self.update(layer_name, "activation", output)
                return hook

            h = module.register_forward_hook(make_hook(name))
            self._handles.append(h)

    # ---- one-shot weight / bias snapshot (call any time; typically once) -
    def collect_weights_and_biases(self, model):
        for name, module in model.named_modules():
            if not isinstance(module, TARGET_LAYER_TYPES):
                continue
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor):
                self.update(name, "weight", w)
            b = getattr(module, "bias", None)
            if isinstance(b, torch.Tensor):
                self.update(name, "bias", b)

    # ---- weight gradients: call after loss.backward() ---------------------
    def collect_weight_gradients(self, model):
        for name, module in model.named_modules():
            if not isinstance(module, TARGET_LAYER_TYPES):
                continue
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor) and w.grad is not None:
                self.update(name, "weight_gradient", w.grad)

    # ---- weight updates: snapshot before optimizer.step(), diff after -----
    def snapshot_weights(self, model):
        snap = {}
        for name, module in model.named_modules():
            if not isinstance(module, TARGET_LAYER_TYPES):
                continue
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor):
                snap[name] = w.detach().clone()
        return snap

    def collect_weight_updates(self, model, pre_snapshot):
        for name, module in model.named_modules():
            if not isinstance(module, TARGET_LAYER_TYPES):
                continue
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor) and name in pre_snapshot:
                delta = w.detach() - pre_snapshot[name]
                self.update(name, "weight_update", delta)

    def remove_forward_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ---- serialization ------------------------------------------------
    def export_summaries(self, max_sample_out=200_000):
        """Returns {layer_name: {tensor_type: stats_dict_with_sample}}"""
        out = {}
        for (layer_name, tensor_type), ts in self.stats.items():
            out.setdefault(layer_name, {})
            out[layer_name][tensor_type] = ts.to_dict_with_sample(max_sample_out)
            out[layer_name][tensor_type]["layer_type"] = self.layer_meta.get(layer_name, {}).get("type")
            out[layer_name][tensor_type]["layer_category"] = self.layer_meta.get(layer_name, {}).get("category")
        return out
