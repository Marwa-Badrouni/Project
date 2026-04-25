"""
models/autoencoder.py — Autoencoder for feature representation learning.

Backend: sklearn MLPRegressor (pure NumPy, no PyTorch required).
For PyTorch, drop-in replacement: same public API, swap internals.

Architecture:
  Encoder: input_dim → hidden_dims → latent_dim
  Decoder: latent_dim → hidden_dims[::-1] → input_dim

Training: iterative partial-fit loop with early stopping on val MSE.
"""

import numpy as np
import pickle
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    AE_HIDDEN_DIMS, AE_LATENT_DIM, AE_EPOCHS,
    AE_LR, AE_BATCH_SIZE, AE_PATIENCE,
    TRAIN_ON_BENIGN_ONLY, MODELS_DIR,
)
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor


class AutoencoderTrainer:
    """
    Trains a two-MLP autoencoder (encoder + decoder) and exposes:
        .fit(X_train, y_train)
        .encode(X)                 → latent embeddings (N, latent_dim)
        .reconstruction_error(X)  → per-sample MSE   (N,)
        .history                  → {"train_loss": [...], "val_loss": [...]}
        .save() / .load()
    """

    def __init__(self, input_dim: int):
        self.input_dim  = input_dim
        self.latent_dim = AE_LATENT_DIM
        self.history    = {"train_loss": [], "val_loss": []}

        enc_hidden = tuple(AE_HIDDEN_DIMS)
        dec_hidden = tuple(reversed(AE_HIDDEN_DIMS))

        # Encoder:  X (input_dim) → latent (latent_dim)
        self._enc = MLPRegressor(
            hidden_layer_sizes=enc_hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=AE_LR,
            batch_size=min(AE_BATCH_SIZE, 512),
            max_iter=1,          # controlled manually per epoch
            warm_start=True,
            tol=0,
            n_iter_no_change=9999,
            random_state=42,
        )
        # Decoder:  latent (latent_dim) → X (input_dim)
        self._dec = MLPRegressor(
            hidden_layer_sizes=dec_hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=AE_LR,
            batch_size=min(AE_BATCH_SIZE, 512),
            max_iter=1,
            warm_start=True,
            tol=0,
            n_iter_no_change=9999,
            random_state=42,
        )
        self._fitted = False

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None):
        if TRAIN_ON_BENIGN_ONLY and y_train is not None:
            X_fit = X_train[y_train == 0]
            print(f"[AE] Training on {len(X_fit):,} benign samples only")
        else:
            X_fit = X_train

        # 90/10 val split
        n_val = max(50, int(len(X_fit) * 0.1))
        idx   = np.random.default_rng(42).permutation(len(X_fit))
        X_tr  = X_fit[idx[:-n_val]]
        X_vl  = X_fit[idx[-n_val:]]

        # Proxy encoder target = first latent_dim principal directions via PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.latent_dim, random_state=42)
        Z_tr_target = pca.fit_transform(X_tr)
        self._pca = pca   # keep for encode fallback if needed

        best_val, patience_cnt = float("inf"), 0
        best_enc, best_dec     = None, None
        epochs = min(AE_EPOCHS, 50)

        print(f"[AE] epochs={epochs}  batch={min(AE_BATCH_SIZE,512)}  "
              f"latent={self.latent_dim}  lr={AE_LR}")

        for epoch in range(1, epochs + 1):
            # ── Encoder step ──────────────────────────────────────────────
            self._enc.max_iter = epoch
            self._enc.fit(X_tr, Z_tr_target)

            Z_tr = self._enc.predict(X_tr)

            # ── Decoder step ──────────────────────────────────────────────
            self._dec.max_iter = epoch
            self._dec.fit(Z_tr, X_tr)
            self._fitted = True

            # ── Losses ────────────────────────────────────────────────────
            Z_vl  = self._enc.predict(X_vl)
            vl_re = self._dec.predict(Z_vl)
            tr_re = self._dec.predict(Z_tr)

            tr_loss = float(np.mean((X_tr - tr_re) ** 2))
            vl_loss = float(np.mean((X_vl - vl_re) ** 2))
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs}  "
                      f"train={tr_loss:.5f}  val={vl_loss:.5f}")

            # ── Early stopping ────────────────────────────────────────────
            if vl_loss < best_val - 1e-8:
                best_val      = vl_loss
                patience_cnt  = 0
                best_enc = pickle.dumps(self._enc)
                best_dec = pickle.dumps(self._dec)
            else:
                patience_cnt += 1
                if patience_cnt >= AE_PATIENCE:
                    print(f"  [AE] Early stopping at epoch {epoch}")
                    break

        if best_enc:
            self._enc = pickle.loads(best_enc)
            self._dec = pickle.loads(best_dec)

        print(f"[AE] Training complete. Best val MSE: {best_val:.6f}")
        return self

    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return latent embeddings (N, latent_dim)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self._enc.predict(X)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error (N,)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        Z     = self._enc.predict(X)
        recon = self._dec.predict(Z)
        return np.mean((X - recon) ** 2, axis=1)

    def save(self, name="autoencoder.pkl"):
        path = MODELS_DIR / name
        with open(path, "wb") as f:
            pickle.dump({
                "enc":       self._enc,
                "dec":       self._dec,
                "pca":       getattr(self, "_pca", None),
                "input_dim": self.input_dim,
                "history":   self.history,
            }, f)
        print(f"[AE] Saved → {path}")

    def load(self, name="autoencoder.pkl"):
        path = MODELS_DIR / name
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self._enc      = ckpt["enc"]
        self._dec      = ckpt["dec"]
        self._pca      = ckpt.get("pca")
        self.input_dim = ckpt["input_dim"]
        self.history   = ckpt["history"]
        self._fitted   = True
        print(f"[AE] Loaded ← {path}")
        return self
