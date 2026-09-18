from copy import deepcopy

import numpy as np
import torch


def forecast_loss(outputs, targets):
    if outputs.shape != (*targets.shape, 10):
        raise ValueError("Expected nine quantiles and one independent point head.")
    levels = torch.arange(1, 10, dtype=outputs.dtype, device=outputs.device)/10
    error = targets.unsqueeze(-1)-outputs[..., 1:]
    pinball = torch.maximum(levels*error, (levels-1)*error)
    return pinball.mean()


def fit(model, train_pair, val_pair, node_weights, settings):
    if settings["loss"] != "mean_pinball":
        raise ValueError("Supported loss: mean_pinball.")
    device = next(model.parameters()).device
    weights = torch.as_tensor(node_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([model.prompts], lr=settings["lr"], weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(settings["seed"])
    best, stale, history, best_state = float("inf"), 0, [], None
    size = settings["batch_size"]
    for epoch in range(settings["epochs"]):
        model.train()
        train_losses = []
        for ids in np.array_split(rng.permutation(len(train_pair[0])), max(1, int(np.ceil(len(train_pair[0])/size)))):
            x, y = (torch.as_tensor(a[ids], dtype=torch.float32, device=device) for a in train_pair)
            optimizer.zero_grad(set_to_none=True)
            loss = forecast_loss(model(x, weights, y.shape[1]), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            train_losses.append((float(loss.detach()), len(ids)))
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for start in range(0, len(val_pair[0]), size):
                x, y = (torch.as_tensor(a[start:start+size], dtype=torch.float32, device=device) for a in val_pair)
                loss = forecast_loss(model(x, weights, y.shape[1]), y)
                val_losses.append((float(loss), len(x)))
        train_mean = sum(v*n for v, n in train_losses)/sum(n for _, n in train_losses)
        val_mean = sum(v*n for v, n in val_losses)/sum(n for _, n in val_losses)
        if not np.isfinite(val_mean):
            raise FloatingPointError("Nonfinite validation loss.")
        history.append({"epoch": epoch+1, "train_loss": train_mean, "val_loss": val_mean})
        print(f"epoch={epoch+1} train={train_mean:.6f} val={val_mean:.6f}")
        if val_mean < best:
            best, stale = val_mean, 0
            best_state = deepcopy(model.prompts.detach().cpu())
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    with torch.no_grad():
        model.prompts.copy_(best_state.to(device))
    return history
