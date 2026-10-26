from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_preprocessor(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> dict:
    mean = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weight))
    scale[scale < 1e-8] = 1.
    y_mean = float(np.average(y, weights=weight))
    y_scale = max(float(np.sqrt(np.average((y-y_mean)**2, weights=weight))), 1e-8)
    return {"x_mean": mean.tolist(), "x_scale": scale.tolist(), "y_mean": y_mean, "y_scale": y_scale}


class MLP(nn.Module):
    def __init__(self, hidden: list[int], dropout: float = 0.):
        super().__init__()
        layers = []
        previous = 5
        for dimension in hidden:
            layers.extend([nn.Linear(previous, dimension), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = dimension
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RawInputModel(nn.Module):
    def __init__(self, network: nn.Module, preprocessing: dict):
        super().__init__()
        self.network = network
        for name in ["x_mean", "x_scale", "y_mean", "y_scale"]:
            self.register_buffer(name, torch.tensor(preprocessing[name], dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network((features-self.x_mean)/self.x_scale) * self.y_scale + self.y_mean


def train_network(x: np.ndarray, y: np.ndarray, weight: np.ndarray, xv: np.ndarray, yv: np.ndarray,
                  wv: np.ndarray, preprocessing: dict, candidate: dict, settings: dict, seed: int):
    set_seed(seed)
    model = MLP(candidate["hidden"], candidate["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    mean, scale = np.array(preprocessing["x_mean"]), np.array(preprocessing["x_scale"])
    xt = torch.tensor((x-mean)/scale, dtype=torch.float32)
    yt = torch.tensor((y-preprocessing["y_mean"])/preprocessing["y_scale"], dtype=torch.float32).reshape(-1, 1)
    wt = torch.tensor(weight/weight.mean(), dtype=torch.float32).reshape(-1, 1)
    validation = torch.tensor((xv-mean)/scale, dtype=torch.float32)
    best, best_state, stale = float("inf"), None, 0
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(settings["max_epochs"]):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total = 0.
        for indices in order.split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = (wt[indices] * (model(xt[indices])-yt[indices])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            predicted = model(validation).numpy().reshape(-1)*preprocessing["y_scale"]+preprocessing["y_mean"]
        mae = float(np.average(np.abs(predicted-yv), weights=wv))
        history.append({"epoch": epoch+1, "train_weighted_mse_standardized": total/len(x), "validation_weighted_mae": mae})
        if mae < best - 1e-5:
            best, best_state, stale = mae, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best
