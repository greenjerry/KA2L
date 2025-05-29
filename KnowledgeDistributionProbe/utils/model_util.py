import torch


def hook_model(model, layernames: list, hook_fn):
    for layername in layernames:
        layer = get_module(model, layername)
        layer.register_forward_hook(hook_fn)


class NetHook:
    def __init__(self, model, layernames: list):
        self.model = model
        self.layernames = layernames
        self.inputs = dict()
        self.outputs = dict()
        self.modules = dict()
        self.module_name_map = dict()

        def hook_fn(module, input, output):
            layername = self.module_name_map[module]
            self.outputs[layername] = recursive_copy(output, clone=True, detach=False, retain_grad=False)
            self.inputs[layername] = recursive_copy(input, clone=True, detach=False, retain_grad=False)

        self.hook_fn = hook_fn

    def __enter__(self):
        for layername in self.layernames:
            layer = get_module(self.model, layername)
            self.modules[layername] = layer.register_forward_hook(self.hook_fn)
            self.module_name_map[layer] = layername
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layername, module in self.modules.items():
            module.remove()


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def get_layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    # Baichuan2-7B-Chat
    if hasattr(model, "model"):
        if kind == "embed":
            return "model.embed_tokens"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def mean_pool(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs
