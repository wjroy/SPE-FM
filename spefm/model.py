# Parts of the masked patching and layer forwarding follow TimesFM.
# Copyright 2025 Google LLC. Licensed under Apache-2.0; see licenses/TimesFM.txt.
import torch
from torch import nn
from torch.nn import functional as F


def load_backbone(weights_path=None):
    from .weights import resolve_backbone_weights
    weights_path = resolve_backbone_weights(weights_path)
    from safetensors.torch import load_file
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    backbone = TimesFM_2p5_200M_torch_module()
    backbone.load_state_dict(load_file(str(weights_path)), strict=True)
    return backbone


class SPEFM(nn.Module):
    def __init__(self, backbone, num_prototypes, prompt_length, injection_layers):
        super().__init__()
        self.backbone = backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.injection_layers = frozenset(injection_layers)
        if not self.injection_layers or min(self.injection_layers) < 0 or max(self.injection_layers) >= backbone.x:
            raise ValueError("Injection layers are zero-based backbone layer indices.")
        if num_prototypes < 1 or prompt_length < 1:
            raise ValueError("Prototype count and prompt length must be positive.")
        reference = next(backbone.parameters())
        self.prompts = nn.Parameter(torch.randn(
            num_prototypes, prompt_length, backbone.md,
            device=reference.device, dtype=reference.dtype) * .02)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, context, node_weights, horizon, *, correct_crossing=False):
        batch, length, nodes = context.shape
        b = self.backbone
        if not 1 <= horizon <= b.o or not 1 <= length <= b.p:
            raise ValueError("The model supports one input patch and one output patch.")
        if node_weights.shape != (nodes, self.prompts.shape[0]):
            raise ValueError("Node/prototype dimensions do not match prompt weights.")
        if not torch.isfinite(context).all() or not torch.isfinite(node_weights).all():
            raise ValueError("Context and association weights must be finite.")
        if torch.any(node_weights < 0) or torch.any(node_weights.sum(-1) <= 0):
            raise ValueError("Association weights must have positive row sums.")
        x = context.permute(0, 2, 1).reshape(batch*nodes, length)
        mean = x.mean(-1, keepdim=True)
        variance = (x.square().mean(-1, keepdim=True)-mean.square()).clamp_min(0)
        scale = (variance+1e-6).sqrt()+1e-6
        normalized = F.pad((x-mean)/scale, (b.p-length, 0)).unsqueeze(1)
        mask = F.pad(torch.zeros_like(x, dtype=torch.bool), (b.p-length, 0), value=True).unsqueeze(1)
        hidden = b.tokenizer(torch.cat((normalized, mask.to(x.dtype)), dim=-1))
        base_mask = mask.all(dim=-1)
        w = node_weights.to(x)
        w = w / w.sum(-1, keepdim=True)
        w = w.unsqueeze(0).expand(batch, -1, -1).reshape(batch*nodes, -1)
        prompt = torch.einsum("nk,kpd->npd", w, self.prompts)
        for index, layer in enumerate(b.stacked_xf):
            if index in self.injection_layers:
                extended = torch.cat((prompt, hidden), dim=1)
                prompt_mask = torch.zeros(prompt.shape[:2], dtype=torch.bool, device=x.device)
                hidden, _ = layer(extended, torch.cat((prompt_mask, base_mask), dim=1), None)
                hidden = hidden[:, self.prompts.shape[1]:]
            else:
                hidden, _ = layer(hidden, base_mask, None)
        output = b.output_projection_point(hidden)[:, -1]
        output = (output*scale+mean).reshape(batch*nodes, b.o, b.q)[:, :horizon]
        if correct_crossing:
            heads = list(output.unbind(-1))
            for i in (4, 3, 2, 1):
                heads[i] = torch.minimum(heads[i], heads[i+1])
            for i in (6, 7, 8, 9):
                heads[i] = torch.maximum(heads[i], heads[i-1])
            output = torch.stack(heads, dim=-1)
        return output.reshape(batch, nodes, horizon, b.q).permute(0, 2, 1, 3)
