"""Compara estrategias y marcos de tiempo con split train/test (60/40).
Los parametros se eligen en train y se reporta solo test. Uso: python research.py [--lev 3]
"""
import argparse, glob, itertools, os
import numpy as np
import pandas as pd

FEE, SLIP = 0.0006, 0.0002


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rsi(c, n):
    d = pd.Series(c).diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return (100 - 100 / (1 + up / dn)).values


def atr(df, n=14):
    c = df.close
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n).mean().values


# Cada estrategia devuelve array de senales {1,-1,0} evaluadas al cierre de cada vela.
def s_ema_trend(df, fast, slow):
    c = df.close.values
    f, s, t = ema(c, fast), ema(c, slow), ema(c, 200)
    sig = np.zeros(len(c))
    sig[(f > s) & (np.r_[False, f[:-1] <= s[:-1]]) & (c > t)] = 1
    sig[(f < s) & (np.r_[False, f[:-1] >= s[:-1]]) & (c < t)] = -1
    return sig


def s_bb_revert(df, n, k):
    c = df.close
    m, sd = c.rolling(n).mean(), c.rolling(n).std()
    sig = np.zeros(len(c))
    sig[(c < m - k * sd).values] = 1
    sig[(c > m + k * sd).values] = -1
    return sig


def s_rsi_revert(df, n, lo):
    r = rsi(df.close.values, n)
    t = ema(df.close.values, 200)
    c = df.close.values
    sig = np.zeros(len(c))
    sig[(r < lo) & (c > t)] = 1
    sig[(r > 100 - lo) & (c < t)] = -1
    return sig


def s_donchian(df, n):
    hi, lo = df.high.rolling(n).max().shift(), df.low.rolling(n).min().shift()
    sig = np.zeros(len(df))
    sig[(df.close > hi).values] = 1
    sig[(df.close < lo).values] = -1
    return sig


STRATS = {
    "ema_trend": (s_ema_trend, [(5, 9, 12), (21, 26, 50)]),
    "bb_revert": (s_bb_revert, [(20, 50), (2.0, 2.5, 3.0)]),
    "rsi_revert": (s_rsi_revert, [(2, 7, 14), (10, 20, 30)]),
    "donchian": (s_donchian, [(20, 50, 100)]),
}


def simulate(df, sig, a, lev, sl, tp, max_hold, equity=100.0, risk=0.02):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    pnls, i, n = [], 1, len(df)
    while i < n - 1:
        if sig[i] == 0 or np.isnan(a[i]) or equity <= 1:
            i += 1
            continue
        side = sig[i]
        entry = o[i + 1] * (1 + side * SLIP)  # entra en la apertura siguiente
        dist = sl * a[i]
        margin = min(equity, equity * risk / (dist / entry * lev))
        stop, take = entry - side * dist, entry + side * tp * a[i]
        liq = entry * (1 - side * 0.9 / lev)
        exit_p, j = c[min(i + 1 + max_hold, n - 1)], min(i + 1 + max_hold, n - 1)
        for k in range(i + 1, min(i + 2 + max_hold, n)):
            hit_sl = l[k] <= stop if side == 1 else h[k] >= stop
            hit_tp = h[k] >= take if side == 1 else l[k] <= take
            liqd = l[k] <= liq if side == 1 else h[k] >= liq
            if hit_sl or liqd:
                exit_p, j = (liq if liqd and not hit_sl else stop), k
                break
            if hit_tp:
                exit_p, j = take, k
                break
        ret = side * (exit_p / entry - 1) - 2 * (FEE + SLIP)
        pnl = max(-margin, margin * lev * ret)
        equity += pnl
        pnls.append(pnl)
        i = j + 1
    return np.array(pnls), equity


def evaluate(df, name, params, lev, rr):
    fn = STRATS[name][0]
    sig = fn(df, *params)
    sig[:250] = 0
    a = atr(df)
    sl, tp = rr
    return simulate(df, sig, a, lev, sl, tp, max_hold=40)


def summarize(p, eq):
    if len(p) < 10:
        return None
    return dict(n=len(p), win=(p > 0).mean(), eq=eq, pf=p[p > 0].sum() / max(1e-9, -p[p <= 0].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lev", type=float, default=3)
    lev = ap.parse_args().lev
    rows = []
    for path in sorted(glob.glob("data/*.csv")):
        pair, tf = os.path.basename(path)[:-4].rsplit("_", 1)
        df = pd.read_csv(path)
        if len(df) < 1000:
            continue
        cut = int(len(df) * 0.6)
        train, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
        for name, (fn, grids) in STRATS.items():
            best = None
            for params in itertools.product(*grids):
                for rr in ((1, 1.5), (1.5, 1.5), (1, 3)):
                    p, eq = evaluate(train, name, params, lev, rr)
                    s = summarize(p, eq)
                    if s and (best is None or s["eq"] > best[0]):
                        best = (s["eq"], params, rr)
            if not best:
                continue
            p, eq = evaluate(test, name, best[1], lev, best[2])
            s = summarize(p, eq)
            if s:
                rows.append(dict(pair=pair, tf=tf, strat=name, params=best[1], sl_tp=best[2],
                                 train_eq=round(best[0], 1), test_n=s["n"], test_win=f"{s['win']:.0%}",
                                 test_pf=round(s["pf"], 2), test_eq=round(s["eq"], 1)))
    out = pd.DataFrame(rows).sort_values("test_eq", ascending=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(out.to_string(index=False))
    print("\nTest con equity final > 100:", int((out.test_eq > 100).sum()), "de", len(out))


if __name__ == "__main__":
    main()
