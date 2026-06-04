"""TCN Agent — extracted from 06_tcn_omni_0fee_v0.

Architecture: Causal TCN (4×64 channels, dilations 1/2/4/8) with dual heads:
  - Direction: 3-class softmax  [P(Down), P(Up), P(Flat)]
  - Volatility: Huber regression (auxiliary multi-task target)

Training strategy: train ONCE on data before ``train_end``; inference on OOS
via sliding-window forward pass (no retraining during OOS period).

Output: pd.DataFrame with columns ``tcn_p_up`` and ``tcn_p_down``, indexed to
the OOS DatetimeIndex (first valid bar is SEQ_LEN bars after oos_start).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Feature sets (mirrors 06_ notebook) ──────────────────────────────────────
LGBM_CORE: list[str] = [
    "stoch_k_14", "ret_2h", "rsi_divergence", "close_vs_sma_7",
    "bear_streak", "close_vs_s1", "macd_hist_5_13", "ad_z_48h", "ret_3h",
]
V1_EXTRA: list[str] = [
    "ret_1h", "rsi_14", "vol_ratio_24h", "bb_position_20",
    "hour_sin", "hour_cos", "atr_14_pct", "hurst_168h",
    "trend_score", "close_vs_sma_50", "ma_bull_score",
]
V4_FEATURES: list[str] = [
    "close_vs_true_vwap", "hurst_24h", "hurst_72h",
    "tfi_pct", "tfi_z_24h", "bb_width_pct", "sideways_flag",
]
STRUCT_FEATURES: list[str] = [
    "liq_vwap_dev_24h", "volat_atr_20_pct", "mtf_alignment", "mtf_h4_rsi",
]
DEFAULT_FEATURES: list[str] = LGBM_CORE + V1_EXTRA + V4_FEATURES + STRUCT_FEATURES


# ── TCN architecture (exact match to 06_ notebook) ────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.pad)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, : -self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel, dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.net(x) + res)


class TCNMultiTask(nn.Module):
    """Causal TCN with dual heads: 3-class direction + vol regression."""

    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            blocks.append(TCNBlock(in_ch, out_ch, kernel, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        d = channels[-1]
        self.head_dir = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 32), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(32, 3),
        )
        self.head_vol = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.tcn(x.transpose(1, 2))   # (B, F, T)
        last = out[:, :, -1]                 # last timestep  (B, C)
        return self.head_dir(last), self.head_vol(last).squeeze(-1)


# ── Fractional differentiation (FFD) ─────────────────────────────────────────

def _ffd_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    w: list[float] = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1], dtype=np.float64)


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    weights = _ffd_weights(d, thres)
    width = len(weights)
    vals = series.values.astype(np.float64)
    out = np.full(len(vals), np.nan)
    for i in range(width - 1, len(vals)):
        out[i] = np.dot(weights, vals[i - width + 1 : i + 1])
    return pd.Series(out, index=series.index, name="fracdiff_close")


# ── TCN Agent ─────────────────────────────────────────────────────────────────

@dataclass
class TCNConfig:
    channels: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    kernel: int = 3
    dropout: float = 0.20
    seq_len: int = 48
    epochs: int = 100
    warmup_epochs: int = 5
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 20
    lambda_vol: float = 0.50
    tbm_vol_window: int = 24
    tbm_mult: float = 2.0
    tbm_vert_h: int = 24
    aux_fwd_h: int = 6
    frac_d: float = 0.4
    ffd_thres: float = 1e-4
    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))


class TCNAgent:
    """Causal TCN agent producing OOS P(Up) and P(Down) signals.

    Training is done ONCE on data before ``train_end``. Inference is then
    performed on the OOS period using a sliding window of length ``seq_len``.

    Parameters
    ----------
    cfg:
        ``TCNConfig`` controlling architecture and training hyperparameters.
    device:
        PyTorch device string. Auto-detected (cuda → mps → cpu) if None.
    """

    AGENT_ID = "tcn_v0"
    SIGNAL_COLS = ("tcn_p_up", "tcn_p_down")

    def __init__(self, cfg: TCNConfig | None = None, device: str | None = None) -> None:
        self.cfg = cfg or TCNConfig()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model: TCNMultiTask | None = None
        self.qt: QuantileTransformer | None = None
        self._all_features: list[str] = []

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Add fracdiff_close and return augmented df + feature list."""
        cfg = self.cfg
        log_close = np.log(df["close"])
        df = df.copy()
        df["fracdiff_close"] = frac_diff_ffd(log_close, cfg.frac_d, cfg.ffd_thres)
        all_feats = cfg.features + ["fracdiff_close"]
        missing = [f for f in all_feats if f not in df.columns]
        if missing:
            for f in missing:
                df[f] = 0.0
        return df, all_feats

    def _tbm_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Triple Barrier Method: 0=Down, 1=Up, 2=Flat."""
        cfg = self.cfg
        log_rets = np.log(df["close"]).diff()
        rolling_vol = log_rets.rolling(cfg.tbm_vol_window).std()
        close_vals = df["close"].values
        vol_vals = rolling_vol.values
        n = len(close_vals)
        labels = np.full(n, np.nan)
        for i in range(n - cfg.tbm_vert_h):
            if np.isnan(vol_vals[i]) or vol_vals[i] <= 0:
                continue
            upper = close_vals[i] * (1.0 + cfg.tbm_mult * vol_vals[i])
            lower = close_vals[i] * (1.0 - cfg.tbm_mult * vol_vals[i])
            label = 2
            for j in range(1, cfg.tbm_vert_h + 1):
                px = close_vals[i + j]
                if px >= upper:
                    label = 1
                    break
                elif px <= lower:
                    label = 0
                    break
            labels[i] = label
        return labels

    def _make_sequences(
        self,
        X: np.ndarray,
        y_dir: np.ndarray,
        y_vol: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        cfg = self.cfg
        n = len(X)
        seqs, dirs, vols, wts = [], [], [], []
        for i in range(cfg.seq_len, n):
            if np.isnan(y_dir[i]) or np.isnan(y_vol[i]):
                continue
            seqs.append(X[i - cfg.seq_len : i])
            dirs.append(int(y_dir[i]))
            vols.append(y_vol[i])
            wts.append(weights[i])
        return (
            np.array(seqs, dtype=np.float32),
            np.array(dirs, dtype=np.int64),
            np.array(vols, dtype=np.float32),
            np.array(wts, dtype=np.float32),
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        train_end: pd.Timestamp,
        val_end: pd.Timestamp | None = None,
        verbose: bool = True,
        save_path: Path | None = None,
    ) -> dict:
        """Train the TCN on data in [df.index[0], train_end).

        Parameters
        ----------
        val_end:
            If given, the validation split uses bars in [train_end, val_end).
            Otherwise 20% of the training data is held out for early stopping.
        save_path:
            If provided, save model weights + quantile transformer here.
        """
        cfg = self.cfg
        df, all_feats = self._build_features(df)
        self._all_features = all_feats

        train_mask = df.index < train_end
        if val_end is not None:
            val_mask = (df.index >= train_end) & (df.index < val_end)
        else:
            n_tr = train_mask.sum()
            val_n = max(50, int(n_tr * 0.20))
            tr_idx = np.where(train_mask)[0]
            val_idx_start = tr_idx[-val_n] if len(tr_idx) >= val_n else tr_idx[0]
            val_mask = np.zeros(len(df), dtype=bool)
            val_mask[val_idx_start:tr_idx[-1] + 1] = True
            train_mask = train_mask.copy()
            train_mask[val_idx_start:] = False

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        # TBM labels
        tbm_all = self._tbm_labels(df)
        y_tr_dir = tbm_all[train_mask]
        y_vl_dir = tbm_all[val_mask]

        # Auxiliary vol target
        log_rets = np.log(df["close"]).diff()
        fwd_vol = pd.concat(
            [log_rets.shift(-h) for h in range(1, cfg.aux_fwd_h + 1)], axis=1
        ).std(axis=1)
        y_tr_vol = fwd_vol.values[train_mask].astype(np.float32)
        y_vl_vol = fwd_vol.values[val_mask].astype(np.float32)

        # Sample weights
        atr_pct = df["atr_14_pct"].values if "atr_14_pct" in df.columns else np.ones(len(df))
        med_atr = np.nanmedian(atr_pct)
        weights = np.clip(atr_pct / (med_atr + 1e-12), 0.3, 3.0)
        w_tr = weights[train_mask].astype(np.float32)
        w_vl = weights[val_mask].astype(np.float32)

        # Normalise
        self.qt = QuantileTransformer(
            n_quantiles=min(2000, len(train_df)),
            output_distribution="normal",
            random_state=42,
        )
        X_tr_raw = self.qt.fit_transform(train_df[all_feats].fillna(0).values)
        X_vl_raw = self.qt.transform(val_df[all_feats].fillna(0).values)

        X_tr, y_tr_d, y_tr_v, w_tr = self._make_sequences(
            X_tr_raw.astype(np.float32), y_tr_dir, y_tr_vol, w_tr
        )
        X_vl, y_vl_d, y_vl_v, w_vl = self._make_sequences(
            X_vl_raw.astype(np.float32), y_vl_dir, y_vl_vol, w_vl
        )

        if verbose:
            print(f"[{self.AGENT_ID}] X_train: {X_tr.shape}  X_val: {X_vl.shape}  device: {self.device}")

        # Class weights
        from collections import Counter
        counts = Counter(int(y) for y in y_tr_d if not np.isnan(y))
        total = sum(counts.values())
        class_weights = torch.tensor(
            [total / (3 * counts.get(k, 1)) for k in range(3)],
            dtype=torch.float32,
        ).to(self.device)

        # Model
        self.model = TCNMultiTask(
            input_dim=len(all_feats),
            channels=cfg.channels,
            kernel=cfg.kernel,
            dropout=cfg.dropout,
        ).to(self.device)

        train_ds = TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr_d),
            torch.from_numpy(y_tr_v), torch.from_numpy(w_tr),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_vl), torch.from_numpy(y_vl_d),
            torch.from_numpy(y_vl_v), torch.from_numpy(w_vl),
        )
        train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        def lr_lambda(epoch: int) -> float:
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / cfg.warmup_epochs
            p = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        def _step(batch: list, training: bool) -> tuple[float, float]:
            xb, yb_d, yb_v, wb = [t.to(self.device) for t in batch]
            assert self.model is not None
            logits, vol_pred = self.model(xb)
            ce_per = F.cross_entropy(logits, yb_d, weight=class_weights, reduction="none")
            hub_per = F.huber_loss(vol_pred, yb_v, reduction="none")
            loss = (ce_per * wb).mean() + cfg.lambda_vol * (hub_per * wb).mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            acc = (logits.argmax(1) == yb_d).float().mean().item()
            return loss.item(), acc

        best_vl = float("inf")
        pat_cnt = 0
        best_state: dict | None = None
        history: dict[str, list] = {"tl": [], "vl": [], "ta": [], "va": []}

        for epoch in range(1, cfg.epochs + 1):
            assert self.model is not None
            self.model.train()
            tl = ta = n_tr = 0
            for batch in train_loader:
                l, acc = _step(batch, True)
                bs = len(batch[0])
                tl += l * bs
                ta += acc * bs
                n_tr += bs

            self.model.eval()
            vl = va = n_vl = 0
            with torch.no_grad():
                for batch in val_loader:
                    l, acc = _step(batch, False)
                    bs = len(batch[0])
                    vl += l * bs
                    va += acc * bs
                    n_vl += bs

            tl /= max(n_tr, 1)
            vl /= max(n_vl, 1)
            ta /= max(n_tr, 1)
            va /= max(n_vl, 1)
            history["tl"].append(tl)
            history["vl"].append(vl)
            history["ta"].append(ta)
            history["va"].append(va)
            scheduler.step()

            if vl < best_vl:
                best_vl = vl
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                pat_cnt = 0
            else:
                pat_cnt += 1
                if pat_cnt >= cfg.patience:
                    if verbose:
                        print(f"[{self.AGENT_ID}] Early stop at epoch {epoch}  best_vl={best_vl:.5f}")
                    break

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"[{self.AGENT_ID}] Ep {epoch:>3}  "
                    f"loss: tr={tl:.4f} vl={vl:.4f}  acc: tr={ta:.4f} vl={va:.4f}"
                )

        assert self.model is not None
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "qt": self.qt, "features": all_feats}, save_path)
            if verbose:
                print(f"[{self.AGENT_ID}] Saved → {save_path}")

        return {"best_val_loss": best_vl, "history": history}

    def load(self, path: Path) -> None:
        """Load model weights and quantile transformer from ``path``."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._all_features = ckpt["features"]
        self.qt = ckpt["qt"]
        cfg = self.cfg
        self.model = TCNMultiTask(
            input_dim=len(self._all_features),
            channels=cfg.channels,
            kernel=cfg.kernel,
            dropout=cfg.dropout,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        oos_start: pd.Timestamp,
        batch_size: int = 512,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Slide the trained TCN over the OOS period and return P(Up)/P(Down).

        Returns a DataFrame with columns ``tcn_p_up`` and ``tcn_p_down``,
        indexed from ``oos_start + seq_len`` bars (first bar with a full window).
        The model must be trained first via ``train()`` or loaded via ``load()``.
        """
        if self.model is None or self.qt is None:
            raise RuntimeError("TCNAgent not trained. Call .train() or .load() first.")

        cfg = self.cfg
        df, all_feats = self._build_features(df)

        oos_mask = df.index >= oos_start
        oos_df = df[oos_mask].copy()

        # We need SEQ_LEN bars of context before oos_start
        pre_start_loc = max(0, df.index.searchsorted(oos_start) - cfg.seq_len)
        context_df = df.iloc[pre_start_loc:]
        X_norm = self.qt.transform(context_df[all_feats].fillna(0).values).astype(np.float32)

        seqs, bar_indices = [], []
        context_oos_start = context_df.index.searchsorted(oos_start)

        for i in range(context_oos_start, len(X_norm)):
            if i < cfg.seq_len:
                continue
            seqs.append(X_norm[i - cfg.seq_len : i])
            bar_indices.append(i)

        if not seqs:
            if verbose:
                print(f"[{self.AGENT_ID}] No valid OOS sequences found.")
            return pd.DataFrame(columns=list(self.SIGNAL_COLS))

        X_all = np.array(seqs, dtype=np.float32)
        all_probs: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X_all), batch_size):
                batch = torch.from_numpy(X_all[start : start + batch_size]).to(self.device)
                logits, _ = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        probs_arr = np.concatenate(all_probs, axis=0)   # (N, 3)  [Down, Up, Flat]
        bar_ts = context_df.index[[i for i in bar_indices]]

        result = pd.DataFrame(
            {"tcn_p_up": probs_arr[:, 1], "tcn_p_down": probs_arr[:, 0]},
            index=bar_ts,
        )
        result = result[result.index >= oos_start]

        if verbose:
            print(
                f"[{self.AGENT_ID}] Inference done: {len(result):,} OOS bars  "
                f"mean P(up)={result['tcn_p_up'].mean():.3f}  "
                f"mean P(down)={result['tcn_p_down'].mean():.3f}"
            )
        return result
