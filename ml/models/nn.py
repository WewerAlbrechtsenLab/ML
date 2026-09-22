from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class _MLP(nn.Module):
    def __init__(self, n_features, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier(ClassifierMixin, BaseEstimator):
    """
    Feedforward NN classifier with an sklearn-compatible fit/predict_proba API,
    so it drops into ml.models.training.nested_cross_validate_models and
    ml.models.estimators.build_models like any other model_registry entry.
    """

    def __init__(
        self,
        hidden_dims=(64, 32),
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=200,
        batch_size=64,
        patience=15,
        val_frac=0.15,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)  # avoid thread oversubscription under joblib n_jobs=-1

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.classes_ = np.array([0, 1])
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)

        Xtr, Xval, ytr, yval = train_test_split(
            X, y,
            test_size=self.val_frac,
            stratify=y,
            random_state=self.random_state,
        )

        pos_weight = torch.tensor(
            [(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32
        )

        self.model_ = _MLP(X.shape[1], self.hidden_dims, self.dropout)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        Xval_t, yval_t = torch.from_numpy(Xval), torch.from_numpy(yval)

        rng = np.random.RandomState(self.random_state)
        n = Xtr_t.shape[0]

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for _ in range(self.epochs):
            self.model_.train()
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.model_(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xval_t), yval_t).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def _pos_proba(self, X):
        check_is_fitted(self, "model_")
        X = self.scaler_.transform(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model_(torch.from_numpy(X))).numpy()
        return probs

    def predict_proba(self, X):
        p1 = self._pos_proba(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self._pos_proba(X) >= 0.5).astype(int)
