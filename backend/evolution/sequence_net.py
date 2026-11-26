from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.core.models import DLPrediction


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(np.clip(shifted, -30.0, 30.0))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _conv1d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int = 1) -> np.ndarray:
    """Causal / same-padded 1D convolution.
    x: (B, T, C_in), w: (K, C_in, C_out), b: (C_out,) -> (B, T, C_out).
    Strictly causal left-padding so bar t never sees bar t+1.
    """
    bsz, t_len, _ = x.shape
    k_size, _, c_out = w.shape
    pad_left = (k_size - 1) * dilation
    x_pad = np.pad(x, ((0, 0), (pad_left, 0), (0, 0)), mode="constant")
    out = np.broadcast_to(b, (bsz, t_len, c_out)).copy()
    for k in range(k_size):
        start = k * dilation
        x_slice = x_pad[:, start : start + t_len, :]  # (B, T, C_in)
        out += np.tensordot(x_slice, w[k], axes=([2], [0]))
    return out


def _conv1d_backward(
    dout: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    dilation: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz, t_len, c_in = x.shape
    k_size, _, _ = w.shape
    pad_left = (k_size - 1) * dilation
    x_pad = np.pad(x, ((0, 0), (pad_left, 0), (0, 0)), mode="constant")
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 1))

    for k in range(k_size):
        start = k * dilation
        x_slice = x_pad[:, start : start + t_len, :]
        dw[k] = np.tensordot(x_slice, dout, axes=([0, 1], [0, 1]))
        dx_pad[:, start : start + t_len, :] += np.tensordot(dout, w[k], axes=([2], [1]))

    dx = dx_pad[:, pad_left:, :]
    return dx, dw, db


class SequenceConvNet:
    """Pure-NumPy Causal 1D-CNN + Residual Sequence Encoder (Python 3.15 native, <1ms CPU inference).

    Architecture:
      Input (B, T=32, C=8)
      -> Causal Conv1D(k=3, C_out=12, dilation=1) + ReLU
      -> Causal Conv1D(k=3, C_out=12, dilation=2) + Residual Skip + ReLU
      -> Global Avg Pool || Global Max Pool -> (B, 24)
      -> Dense(24 -> 16) + ReLU
      -> Head 1: Softmax(3) [P_down, P_flat, P_up]
      -> Head 2: Sigmoid(1) [vol_expansion_prob]
    """

    def __init__(self, in_channels: int = 8, filters: int = 12, hidden: int = 16, seed: int = 42) -> None:
        self.in_channels = in_channels
        self.filters = filters
        self.hidden = hidden
        self.rng = np.random.default_rng(seed)
        self.trained: bool = False
        self.val_loss_history: list[float] = []
        self.last_val_accuracy: float = 0.0
        self.params: dict[str, np.ndarray] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        r = self.rng
        f = self.filters
        h = self.hidden
        self.params = {
            "w1": r.normal(0.0, np.sqrt(2.0 / (3 * self.in_channels)), (3, self.in_channels, f)),
            "b1": np.zeros((f,), dtype=np.float64),
            "w2": r.normal(0.0, np.sqrt(2.0 / (3 * f)), (3, f, f)),
            "b2": np.zeros((f,), dtype=np.float64),
            "wd": r.normal(0.0, np.sqrt(2.0 / (2 * f)), (2 * f, h)),
            "bd": np.zeros((h,), dtype=np.float64),
            "w_dir": r.normal(0.0, np.sqrt(1.0 / h), (h, 3)),
            "b_dir": np.zeros((3,), dtype=np.float64),
            "w_vol": r.normal(0.0, np.sqrt(1.0 / h), (h, 1)),
            "b_vol": np.zeros((1,), dtype=np.float64),
        }

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        p = self.params
        z1 = _conv1d_forward(x, p["w1"], p["b1"], dilation=1)
        a1 = np.maximum(z1, 0.0)
        z2 = _conv1d_forward(a1, p["w2"], p["b2"], dilation=2) + a1  # Residual skip
        a2 = np.maximum(z2, 0.0)

        avg_pool = np.mean(a2, axis=1)  # (B, F)
        max_idx = np.argmax(a2, axis=1)  # (B, F)
        max_pool = np.max(a2, axis=1)  # (B, F)
        pooled = np.concatenate([avg_pool, max_pool], axis=1)  # (B, 2F)

        zd = pooled @ p["wd"] + p["bd"]
        ad = np.maximum(zd, 0.0)

        dir_logits = ad @ p["w_dir"] + p["b_dir"]
        dir_probs = _softmax(dir_logits)
        vol_logit = ad @ p["w_vol"] + p["b_vol"]
        vol_prob = _sigmoid(vol_logit).squeeze(-1)

        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "max_idx": max_idx,
            "pooled": pooled,
            "zd": zd,
            "ad": ad,
            "dir_probs": dir_probs,
            "vol_prob": vol_prob,
        }
        return dir_probs, vol_prob, cache

    def _compute_loss_and_grads(
        self,
        x: np.ndarray,
        y_dir: np.ndarray,
        y_vol: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        p = self.params
        bsz = x.shape[0]
        dir_probs, vol_prob, c = self._forward(x)

        eps = 1e-9
        ce_loss = -float(np.mean(np.log(np.clip(dir_probs[np.arange(bsz), y_dir], eps, 1.0))))
        bce_loss = -float(
            np.mean(
                y_vol * np.log(np.clip(vol_prob, eps, 1.0))
                + (1.0 - y_vol) * np.log(np.clip(1.0 - vol_prob, eps, 1.0))
            )
        )
        total_loss = ce_loss + 0.35 * bce_loss

        # Gradients from heads
        d_dir_logits = dir_probs.copy()
        d_dir_logits[np.arange(bsz), y_dir] -= 1.0
        d_dir_logits /= bsz

        d_vol_logit = (0.35 * (vol_prob - y_vol) / bsz)[:, None]

        grads: dict[str, np.ndarray] = {}
        grads["w_dir"] = c["ad"].T @ d_dir_logits
        grads["b_dir"] = np.sum(d_dir_logits, axis=0)
        grads["w_vol"] = c["ad"].T @ d_vol_logit
        grads["b_vol"] = np.sum(d_vol_logit, axis=0)

        d_ad = d_dir_logits @ p["w_dir"].T + d_vol_logit @ p["w_vol"].T
        d_zd = d_ad * (c["zd"] > 0.0)
        grads["wd"] = c["pooled"].T @ d_zd
        grads["bd"] = np.sum(d_zd, axis=0)

        d_pooled = d_zd @ p["wd"].T  # (B, 2F)
        f = self.filters
        t_len = x.shape[1]
        d_avg = d_pooled[:, :f]
        d_max = d_pooled[:, f:]

        d_a2 = np.broadcast_to(d_avg[:, None, :] / t_len, (bsz, t_len, f)).copy()
        b_idx = np.arange(bsz)[:, None]
        f_idx = np.arange(f)[None, :]
        d_a2[b_idx, c["max_idx"], f_idx] += d_max

        d_z2 = d_a2 * (c["z2"] > 0.0)
        d_a1_conv, grads["w2"], grads["b2"] = _conv1d_backward(d_z2, c["a1"], p["w2"], dilation=2)
        d_a1 = d_a1_conv + d_z2  # Residual skip gradient
        d_z1 = d_a1 * (c["z1"] > 0.0)
        _, grads["w1"], grads["b1"] = _conv1d_backward(d_z1, x, p["w1"], dilation=1)

        return total_loss, grads

    def fit(
        self,
        x: np.ndarray,
        y_dir: np.ndarray,
        y_vol: np.ndarray,
        epochs: int = 18,
        batch_size: int = 32,
        lr: float = 0.008,
        weight_decay: float = 1e-3,
        val_ratio: float = 0.2,
    ) -> dict[str, float]:
        """Train with strict chronological train/val split (zero lookahead), AdamW, cosine LR & early stopping."""
        n = len(x)
        if n < 30:
            return {"train_loss": 0.0, "val_loss": 0.0, "val_acc": 0.0}

        # Chronological split with small purge gap
        split = max(20, int(n * (1.0 - val_ratio)))
        gap = min(4, max(0, n - split - 8))
        x_tr, yd_tr, yv_tr = x[:split], y_dir[:split], y_vol[:split]
        x_val, yd_val, yv_val = x[split + gap :], y_dir[split + gap :], y_vol[split + gap :]
        if len(x_val) < 5:
            x_val, yd_val, yv_val = x_tr[-10:], yd_tr[-10:], yv_tr[-10:]

        # Reset AdamW moments
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_mom = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        best_val_loss = float("inf")
        best_params = {k: val.copy() for k, val in self.params.items()}
        self.val_loss_history = []
        last_train_loss = 0.0
        patience = 5
        stale = 0

        for ep in range(epochs):
            # Cosine LR schedule
            cur_lr = lr * 0.5 * (1.0 + float(np.cos(np.pi * ep / max(epochs, 1))))
            cur_lr = max(cur_lr, lr * 0.1)

            perm = self.rng.permutation(len(x_tr))
            ep_losses: list[float] = []
            for start in range(0, len(x_tr), batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 4:
                    continue
                loss, grads = self._compute_loss_and_grads(x_tr[idx], yd_tr[idx], yv_tr[idx])
                ep_losses.append(loss)
                step += 1
                for k, g in grads.items():
                    g_clip = np.clip(g, -2.0, 2.0)
                    m[k] = beta1 * m[k] + (1.0 - beta1) * g_clip
                    v_mom[k] = beta2 * v_mom[k] + (1.0 - beta2) * (g_clip ** 2)
                    m_hat = m[k] / (1.0 - beta1 ** step)
                    v_hat = v_mom[k] / (1.0 - beta2 ** step)
                    # Decoupled weight decay for weight matrices only
                    if k.startswith("w"):
                        self.params[k] *= 1.0 - cur_lr * weight_decay
                    self.params[k] -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

            last_train_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            val_loss, _ = self._compute_loss_and_grads(x_val, yd_val, yv_val)
            self.val_loss_history.append(round(val_loss, 4))

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_params = {k: val.copy() for k, val in self.params.items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.params = best_params
        val_dir_probs, _, _ = self._forward(x_val)
        self.last_val_accuracy = float(np.mean(np.argmax(val_dir_probs, axis=1) == yd_val))
        self.trained = True
        return {
            "train_loss": round(last_train_loss, 4),
            "val_loss": round(best_val_loss, 4),
            "val_acc": round(self.last_val_accuracy, 4),
        }

    def predict(self, seq: np.ndarray | None) -> DLPrediction:
        if seq is None:
            return DLPrediction(trained=self.trained)
        x = seq[None, :, :] if seq.ndim == 2 else seq
        dir_p, vol_p, _ = self._forward(x)
        p0, p1, p2 = float(dir_p[0, 0]), float(dir_p[0, 1]), float(dir_p[0, 2])
        vp = float(vol_p[0])
        return DLPrediction(
            prob_down=round(p0, 4),
            prob_flat=round(p1, 4),
            prob_up=round(p2, 4),
            vol_expansion=round(vp, 4),
            trained=self.trained,
        )

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            **self.params,
            meta_trained=np.array([1 if self.trained else 0]),
            meta_val_acc=np.array([self.last_val_accuracy]),
            meta_val_hist=np.array(self.val_loss_history or [0.0]),
        )

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = np.load(p)
        for k in list(self.params.keys()):
            if k in data:
                self.params[k] = data[k].astype(np.float64)
        self.trained = bool(int(data["meta_trained"][0])) if "meta_trained" in data else True
        self.last_val_accuracy = float(data["meta_val_acc"][0]) if "meta_val_acc" in data else 0.0
        self.val_loss_history = [float(v) for v in data["meta_val_hist"]] if "meta_val_hist" in data else []
        return True
