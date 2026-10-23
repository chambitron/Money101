"""Backtest de scalping 3m: cruce EMA9/21 a favor de EMA200, filtro RSI, SL/TP por ATR.
Incluye comisiones y slippage. Uso: python backtest.py data/BTC-USDT_3m.csv [--lev 5]
"""
import argparse
import numpy as np
import pandas as pd

FEE = 0.0006   # taker por lado
SLIP = 0.0002  # slippage por lado


def indicators(df):
    c = df.close
    df["e9"], df["e21"], df["e200"] = (c.ewm(span=s, adjust=False).mean() for s in (9, 21, 200))
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1/14).mean(), (-d.clip(upper=0)).ewm(alpha=1/14).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14).mean()
    return df


def run(df, lev=5.0, risk=0.02, sl_atr=1.0, tp_atr=1.5, equity=100.0, daily_loss=0.10):
    df = indicators(df.copy()).dropna().reset_index(drop=True)
    trades, pos, day, day_start = [], None, None, equity
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        d = pd.to_datetime(r.ts, unit="ms").date()
        if d != day:
            day, day_start = d, equity
        if pos:
            hit_sl = r.low <= pos["sl"] if pos["side"] == 1 else r.high >= pos["sl"]
            hit_tp = r.high >= pos["tp"] if pos["side"] == 1 else r.low <= pos["tp"]
            liq = r.low <= pos["liq"] if pos["side"] == 1 else r.high >= pos["liq"]
            if hit_sl or hit_tp or liq:  # si SL y TP en la misma vela, asumimos SL (conservador)
                exit_p = pos["liq"] if liq and not hit_sl else pos["sl"] if hit_sl else pos["tp"]
                ret = pos["side"] * (exit_p / pos["entry"] - 1) - 2 * (FEE + SLIP)
                pnl = max(-pos["margin"], pos["margin"] * lev * ret)
                equity += pnl
                trades.append(pnl)
                pos = None
            continue
        if equity <= 1 or equity < day_start * (1 - daily_loss):
            continue
        long_ = p.e9 <= p.e21 and r.e9 > r.e21 and r.close > r.e200 and 45 < r.rsi < 70
        short = p.e9 >= p.e21 and r.e9 < r.e21 and r.close < r.e200 and 30 < r.rsi < 55
        if long_ or short:
            side = 1 if long_ else -1
            entry = r.close * (1 + side * SLIP)
            margin = equity * risk / (sl_atr * r.atr / entry * lev)  # riesgo fijo por trade
            margin = min(margin, equity)
            pos = dict(side=side, entry=entry, margin=margin,
                       sl=entry - side * sl_atr * r.atr, tp=entry + side * tp_atr * r.atr,
                       liq=entry * (1 - side * 0.9 / lev))
    t = np.array(trades)
    if not len(t):
        return dict(trades=0)
    curve = 100 + t.cumsum()
    return dict(trades=len(t), win_rate=f"{(t > 0).mean():.1%}", final_equity=round(equity, 2),
                max_drawdown=f"{(1 - curve / np.maximum.accumulate(curve)).max():.1%}",
                avg_win=round(t[t > 0].mean(), 3) if (t > 0).any() else 0,
                avg_loss=round(t[t <= 0].mean(), 3) if (t <= 0).any() else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--lev", type=float, default=5)
    a = ap.parse_args()
    print(run(pd.read_csv(a.csv), lev=a.lev))
