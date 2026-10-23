"""Replica estrategias publicas de scalping de TradingView (EMA9/21 + VWAP + volumen; pullback a EMA21)
y las prueba con el motor de research.py (train 60% / test 40%, costos incluidos)."""
import glob, itertools, os
import numpy as np, pandas as pd
import research as R


def vwap_daily(df):
    d = pd.to_datetime(df.ts, unit="ms").dt.date
    tp = (df.high + df.low + df.close) / 3
    return ((tp * df.volume).groupby(d).cumsum() / df.volume.groupby(d).cumsum().replace(0, np.nan)).values


def s_tv_cross_vwap(df, vol_mult):
    c = df.close.values
    f, s, v = R.ema(c, 9), R.ema(c, 21), vwap_daily(df)
    vol_ok = (df.volume > vol_mult * df.volume.rolling(20).mean()).values
    up = (f > s) & np.r_[False, f[:-1] <= s[:-1]]
    dn = (f < s) & np.r_[False, f[:-1] >= s[:-1]]
    sig = np.zeros(len(c))
    sig[up & (c > v) & vol_ok] = 1
    sig[dn & (c < v) & vol_ok] = -1
    return sig


def s_tv_pullback(df, rsi_max):
    c, lo, hi = df.close.values, df.low.values, df.high.values
    f, s, v = R.ema(c, 9), R.ema(c, 21), vwap_daily(df)
    r = R.rsi(c, 14)
    prev_c = np.r_[c[0], c[:-1]]
    long_ = (f > s) & (c > v) & (lo <= s) & (c > s) & (r < rsi_max)
    short = (f < s) & (c < v) & (hi >= s) & (c < s) & (r > 100 - rsi_max)
    sig = np.zeros(len(c))
    sig[long_] = 1
    sig[short] = -1
    return sig


R.STRATS.clear()
R.STRATS["tv_cross_vwap"] = (s_tv_cross_vwap, [(0.0, 1.0, 1.5)])
R.STRATS["tv_pullback"] = (s_tv_pullback, [(55, 65, 80)])

if __name__ == "__main__":
    rows, lev = [], 3
    for path in sorted(glob.glob("data/*_[35]m.csv")):
        pair, tf = os.path.basename(path)[:-4].rsplit("_", 1)
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(df) < 1000:
            continue
        cut = int(len(df) * 0.6)
        train, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
        for name, (fn, grids) in R.STRATS.items():
            best = None
            for params in itertools.product(*grids):
                for rr in ((1, 1.5), (1.5, 1.5), (1, 3)):
                    p, eq = R.evaluate(train, name, params, lev, rr)
                    s = R.summarize(p, eq)
                    if s and (best is None or s["eq"] > best[0]):
                        best = (s["eq"], params, rr)
            if not best:
                continue
            p, eq = R.evaluate(test, name, best[1], lev, best[2])
            s = R.summarize(p, eq)
            if s:
                rows.append(dict(pair=pair, tf=tf, strat=name, params=best[1], sl_tp=best[2],
                                 train_eq=round(best[0], 1), test_n=s["n"], test_win=f"{s['win']:.0%}",
                                 test_pf=round(s["pf"], 2), test_eq=round(s["eq"], 1)))
    out = pd.DataFrame(rows).sort_values("test_eq", ascending=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\nTest con equity final > 100:", int((out.test_eq > 100).sum()), "de", len(out))
