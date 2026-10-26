from __future__ import annotations

import copy
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..model import MLP, set_seed


class TabularMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(16,), seed=2026, max_epochs=300, patience=25, batch_size=128):
        self.hidden, self.seed = hidden, seed
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size

    def _learn(self, x, y, epochs, validation=None):
        set_seed(self.seed)
        network = MLP(list(self.hidden), .1)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.01)
        tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        best, best_epoch, stale = float("inf"), 1, 0
        for epoch in range(epochs):
            network.train()
            for indices in torch.randperm(len(tx)).split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(network(tx[indices]), ty[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 10.)
                optimizer.step()
            if validation is not None:
                network.eval()
                vx, vy = validation
                with torch.no_grad():
                    score = float(nn.functional.binary_cross_entropy_with_logits(
                        network(torch.tensor(vx, dtype=torch.float32)), torch.tensor(vy, dtype=torch.float32).reshape(-1, 1)))
                if score < best - 1e-5:
                    best, best_epoch, stale = score, epoch + 1, 0
                else:
                    stale += 1
                if stale >= self.patience:
                    break
        return network.eval(), best_epoch

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, int)
        if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
            raise ValueError("Five finite features and both binary classes required")
        train, stop = train_test_split(np.arange(len(y)), test_size=.15, stratify=y, random_state=self.seed)
        inner_scaler = StandardScaler().fit(x[train])
        _, epochs = self._learn(inner_scaler.transform(x[train]), y[train], self.max_epochs,
                               (inner_scaler.transform(x[stop]), y[stop]))
        # Refit all fold-training data only after fixing the epoch count internally.
        self.scaler_ = StandardScaler().fit(x)
        self.network_, _ = self._learn(self.scaler_.transform(x), y, epochs)
        self.epochs_, self.classes_, self.n_features_in_ = epochs, np.array([0, 1]), 5
        return self

    def predict_proba(self, x):
        self.network_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.network_(torch.tensor(self.scaler_.transform(x), dtype=torch.float32))).numpy().reshape(-1)
        return np.column_stack([1-p, p])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= .5).astype(int)


class CalibratedModel:
    def __init__(self, estimator, calibrator):
        self.estimator, self.calibrator = estimator, calibrator

    @staticmethod
    def logits(estimator, x):
        p = np.asarray(estimator.predict_proba(x)[:, 1], float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid estimator probabilities")
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.log(p/(1-p)).reshape(-1, 1)

    def predict_proba(self, x):
        return self.calibrator.predict_proba(self.logits(self.estimator, x))
