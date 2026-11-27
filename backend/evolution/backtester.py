from __future__ import annotations

import math

import numpy as np

from backend.core.config import TradingConfig
from backend.core.models import (
    BacktestMetrics,
    Bar,
    DLPrediction,
    Side,
    SymbolSpec,
    TradeRecord,
)
from backend.evolution.meta_labeler import CalibratedMetaLabeler
from backend.evolution.sequence_net import SequenceConvNet
from backend.features.dataset import build_sequence_tensor, extract_meta_features
from backend.features.regime import classify_regime
from backend.risk.manager import DEFAULT_SPECS
from backend.strategies.registry import StrategySuite


LOOKBACK_BARS = 300


class CausalBacktester:
    """Realistic M1-M5 bar-by-bar backtester with ZERO lookahead bias.

    - Signals evaluated strictly on closed bar i.
    - Entries filled on next bar (i+1) open + Bid/Ask spread + slippage penalty.
    - Intrabar SL/TP evaluated pessimistically (SL checked before TP if both touched).
    - Tracks exact MAE/MFE in R-multiples and applies BE / ATR trailing exits.
    """

    def __init__(
        self,
        slippage_points: float = 2.0,
        initial_equity: float = 10000.0,
    ) -> None:
        self.slippage_points = slippage_points
        self.initial_equity = initial_equity

    def run(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        config: TradingConfig,
        specs: dict[str, SymbolSpec] | None = None,
        dl_net: SequenceConvNet | None = None,
        ml_labeler: CalibratedMetaLabeler | None = None,
        collect_meta_samples: bool = False,
    ) -> tuple[BacktestMetrics, list[TradeRecord], tuple[np.ndarray, np.ndarray]]:
        suite = StrategySuite.from_config(config)
        sym_specs = specs or DEFAULT_SPECS
        trades: list[TradeRecord] = []
        meta_x: list[np.ndarray] = []
        meta_y: list[int] = []

        equity = self.initial_equity
        peak_equity = equity
        max_dd_pct = 0.0
        risk_pct = config.risk.risk_per_trade_pct
        max_sp_atr = config.risk.max_spread_atr_ratio
        be_r = config.ai.rl_exit_be_trigger_r
        trail_r = config.ai.rl_exit_trail_trigger_r

        for symbol, bars in bars_by_symbol.items():
            if len(bars) < 60:
                continue
            spec = sym_specs.get(symbol, DEFAULT_SPECS["XAUUSD"])
            i = 48
            while i < len(bars) - 2:
                # Indicators only look back a few hundred bars (the live engine keeps 600), so a
                # trailing window keeps each step O(1) instead of re-reading days of history.
                window = bars[max(0, i + 1 - LOOKBACK_BARS) : i + 1]
                assess = classify_regime(window)
                if assess is None:
                    i += 1
                    continue

                spread_price = bars[i].spread_points * spec.point
                if (spread_price / max(assess.indicators.atr14, 1e-9)) > max_sp_atr:
                    i += 1
                    continue

                sigs = suite.scan_all(symbol, window, assess.indicators, assess.regime)
                if not sigs:
                    i += 1
                    continue

                # Select strongest signal on bar i
                best_sig = max(sigs, key=lambda s: s.confidence)

                # Causal DL inference
                seq = build_sequence_tensor(window, seq_len=config.ai.dl_seq_len, point=spec.point)
                dl_pred = dl_net.predict(seq) if (dl_net is not None and dl_net.trained) else DLPrediction()
                feat_vec = extract_meta_features(best_sig, assess.indicators, spread_price, dl_pred, bars[i].timestamp)

                # ML Meta-Labeler filter (skip filtering if collecting raw training data)
                ml_prob = ml_labeler.predict_win_probability(feat_vec) if (ml_labeler and ml_labeler.trained) else 0.62
                if not collect_meta_samples and ml_labeler and ml_labeler.trained:
                    if ml_prob < config.ai.ml_min_probability:
                        i += 1
                        continue

                # Fill on NEXT bar open + spread + slippage (strictly causal!)
                fill_bar = bars[i + 1]
                slip = self.slippage_points * spec.point
                if best_sig.side == Side.BUY:
                    entry = fill_bar.open + spread_price + slip
                    risk_dist = max(entry - best_sig.sl_price, assess.indicators.atr14 * 0.5)
                    sl = entry - risk_dist
                    tp = entry + max(best_sig.tp_price - best_sig.entry_price, risk_dist * config.risk.min_rr_ratio)
                else:
                    entry = fill_bar.open - slip
                    risk_dist = max(best_sig.sl_price - entry, assess.indicators.atr14 * 0.5)
                    sl = entry + risk_dist
                    tp = entry - max(best_sig.entry_price - best_sig.tp_price, risk_dist * config.risk.min_rr_ratio)

                # Simulate forward bar-by-bar lifecycle (max 24 bars for M1-M5 scalping)
                cur_sl = sl
                mae_r = 0.0
                mfe_r = 0.0
                exit_price = fill_bar.close
                exit_bar_idx = i + 1
                exit_reason = "TIME_EXPIRY"
                rl_trace = ["HOLD"]

                for j in range(i + 1, min(len(bars), i + 25)):
                    b = bars[j]
                    if best_sig.side == Side.BUY:
                        adv_r = (b.low - entry) / risk_dist
                        fav_r = (b.high - entry) / risk_dist
                        mae_r = min(mae_r, adv_r)
                        mfe_r = max(mfe_r, fav_r)
                        # Pessimistic order: check SL first
                        if b.low <= cur_sl:
                            exit_price = cur_sl - slip * 0.5
                            exit_bar_idx = j
                            exit_reason = "SL" if cur_sl <= sl + 1e-9 else "TRAIL_BE"
                            break
                        if b.high >= tp:
                            exit_price = tp
                            exit_bar_idx = j
                            exit_reason = "TP"
                            break
                        # Dynamic BE / Trailing update at bar close for subsequent bar
                        close_r = (b.close - entry) / risk_dist
                        if close_r >= trail_r:
                            new_sl = b.close - 0.65 * assess.indicators.atr14
                            if new_sl > cur_sl:
                                cur_sl = new_sl
                                if rl_trace[-1] != "TIGHT_TRAIL":
                                    rl_trace.append("TIGHT_TRAIL")
                        elif close_r >= be_r and cur_sl < entry:
                            cur_sl = entry + 0.05 * risk_dist
                            rl_trace.append("MOVE_BE")
                    else:
                        adv_r = (entry - b.high) / risk_dist
                        fav_r = (entry - b.low) / risk_dist
                        mae_r = min(mae_r, adv_r)
                        mfe_r = max(mfe_r, fav_r)
                        if b.high + spread_price >= cur_sl:
                            exit_price = cur_sl + slip * 0.5
                            exit_bar_idx = j
                            exit_reason = "SL" if cur_sl >= sl - 1e-9 else "TRAIL_BE"
                            break
                        if b.low + spread_price <= tp:
                            exit_price = tp
                            exit_bar_idx = j
                            exit_reason = "TP"
                            break
                        close_r = (entry - (b.close + spread_price)) / risk_dist
                        if close_r >= trail_r:
                            new_sl = b.close + spread_price + 0.65 * assess.indicators.atr14
                            if new_sl < cur_sl:
                                cur_sl = new_sl
                                if rl_trace[-1] != "TIGHT_TRAIL":
                                    rl_trace.append("TIGHT_TRAIL")
                        elif close_r >= be_r and cur_sl > entry:
                            cur_sl = entry - 0.05 * risk_dist
                            rl_trace.append("MOVE_BE")
                    exit_price = b.close
                    exit_bar_idx = j

                raw_pnl_r = (
                    (exit_price - entry) / risk_dist
                    if best_sig.side == Side.BUY
                    else (entry - exit_price) / risk_dist
                )
                # Deduct realistic commission impact in R-units (~0.04R per round trip on scalping)
                comm_r = 0.04
                net_r = raw_pnl_r - comm_r
                risk_money = equity * (risk_pct / 100.0)
                pnl_money = risk_money * net_r
                equity = max(100.0, equity + pnl_money)
                peak_equity = max(peak_equity, equity)
                dd = ((peak_equity - equity) / peak_equity) * 100.0
                max_dd_pct = max(max_dd_pct, dd)

                dl_align = dl_pred.prob_up if best_sig.side == Side.BUY else dl_pred.prob_down
                trades.append(
                    TradeRecord(
                        is_shadow=True,
                        version_id=config.app.version,
                        symbol=symbol,
                        strategy_id=best_sig.strategy_id,
                        regime=best_sig.regime,
                        side=best_sig.side,
                        volume=round(max(spec.volume_min, risk_money / 100.0), 2),
                        entry_price=round(entry, spec.digits),
                        exit_price=round(exit_price, spec.digits),
                        sl_price=round(sl, spec.digits),
                        tp_price=round(tp, spec.digits),
                        open_time=fill_bar.timestamp,
                        close_time=bars[exit_bar_idx].timestamp,
                        pnl_money=round(pnl_money, 2),
                        pnl_r=round(net_r, 4),
                        mae_r=round(mae_r, 4),
                        mfe_r=round(mfe_r, 4),
                        spread_at_entry=bars[i].spread_points,
                        slippage_points=self.slippage_points,
                        commission=round(risk_money * comm_r, 2),
                        ml_confidence=ml_prob,
                        dl_confidence=round(dl_align, 4),
                        exit_reason=exit_reason,
                        rl_exit_trace=" -> ".join(rl_trace),
                    )
                )
                if collect_meta_samples:
                    meta_x.append(feat_vec)
                    meta_y.append(1 if net_r > 0 else 0)

                # Advance past exit bar to prevent overlapping position distortion
                i = exit_bar_idx + 1

        metrics = self._summarize_metrics(trades, equity, max_dd_pct)
        x_arr = np.stack(meta_x, axis=0) if meta_x else np.empty((0, 24), dtype=np.float64)
        y_arr = np.array(meta_y, dtype=np.int64) if meta_y else np.empty((0,), dtype=np.int64)
        return metrics, trades, (x_arr, y_arr)

    def _summarize_metrics(
        self,
        trades: list[TradeRecord],
        final_equity: float,
        max_dd_pct: float,
    ) -> BacktestMetrics:
        n = len(trades)
        if n == 0:
            return BacktestMetrics()
        rs = np.array([t.pnl_r for t in trades], dtype=np.float64)
        wins = rs[rs > 0]
        losses = -rs[rs <= 0]
        gross_win = float(np.sum(wins))
        gross_loss = float(np.sum(losses))
        pf = gross_win / gross_loss if gross_loss > 1e-9 else (3.0 if gross_win > 0 else 0.0)
        win_rate = float(np.mean(rs > 0))
        expectancy = float(np.mean(rs))
        std_r = float(np.std(rs))
        sharpe = (expectancy / std_r) * math.sqrt(min(n, 252)) if std_r > 1e-9 else 0.0
        net_ret = ((final_equity - self.initial_equity) / self.initial_equity) * 100.0

        return BacktestMetrics(
            total_trades=n,
            win_rate=round(win_rate, 4),
            profit_factor=round(min(pf, 9.99), 4),
            sharpe_ratio=round(sharpe, 4),
            expectancy_r=round(expectancy, 4),
            max_drawdown_pct=round(max_dd_pct, 4),
            net_return_pct=round(net_ret, 4),
            avg_mae_r=round(float(np.mean([t.mae_r for t in trades])), 4),
            avg_mfe_r=round(float(np.mean([t.mfe_r for t in trades])), 4),
        )
