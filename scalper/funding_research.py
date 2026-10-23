"""Estrategias de funding rate sobre datos publicos de BloFin.

A) Carry delta-neutral (spot largo + perp corto): solo cobra funding, con costos de entrada/salida
   de ambas patas. Ignora riesgo de base y requiere cuenta spot fuera de BloFin.
B) Direccional contrarian/momentum: funding extremo -> posicion perp N horas, con apalancamiento.
   Parametros elegidos en train (60%), reportado solo test (40%).
Uso: python funding_research.py [--lev 3]
"""
import argparse, itertools
import numpy as np
import pandas as pd

PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
PERP_FEE, SPOT_FEE, SLIP = 0.0006, 0.001, 0.0002


def load(pair):
    f = pd.read_csv(f"data/{pair}_funding.csv")
    c = pd.read_csv(f"data/long/{pair}_1H.csv")
    c["ts"] = c.ts.astype("int64")
    px = c.set_index("ts").close
    # funding se paga a ts; la senal se conoce en ese instante y se opera al precio de ese cierre horario
    f = f[f.ts.isin(px.index)].copy()
    f["px"] = px.loc[f.ts].values
    return f.reset_index(drop=True), px


def carry(f, thr):
    """Abre carry cuando el funding supera thr y cierra cuando baja de 0. Retorno sobre el nocional."""
    eq, open_, pnl_ev, n_trades, paid = 0.0, False, [], 0, 0.0
    for r in f.rate.values:
        if not open_ and r > thr:
            open_, n_trades = True, n_trades + 1
            eq -= PERP_FEE + SLIP + SPOT_FEE + SLIP  # costo de entrada, ambas patas
        if open_:
            eq += r
            paid += r
            if r < 0:
                open_ = False
                eq -= PERP_FEE + SLIP + SPOT_FEE + SLIP
    days = len(f) * 8 / 24
    return dict(ret=eq, apr=eq / days * 365, trades=n_trades, days=days)


def directional(f, px, lev, thr, hold, mode, capital=100.0, risk_frac=1.0):
    """mode=+1 contrarian (short si funding alto), -1 momentum. Mantiene hold horas, cobra/paga funding."""
    ts, rate = f.ts.values, f.rate.values
    eq, pnls, i = capital, [], 0
    idx = px.index.values
    while i < len(f):
        r = rate[i]
        if abs(r) < thr:
            i += 1
            continue
        side = -mode * np.sign(r)
        t0 = ts[i]
        t1 = t0 + hold * 3_600_000
        j = np.searchsorted(idx, t1)
        if j >= len(idx):
            break
        entry = px.loc[t0] * (1 + side * SLIP)
        exit_ = px.iloc[j] * (1 - side * SLIP)
        seg = px.iloc[np.searchsorted(idx, t0):j + 1]
        liq = entry * (1 - side * 0.9 / lev)
        liqd = (seg.min() <= liq) if side == 1 else (seg.max() >= liq)
        pay = rate[(ts > t0) & (ts <= t1)].sum()  # funding pagado/cobrado durante la tenencia
        ret = -1.0 if liqd else side * (exit_ / entry - 1) - 2 * PERP_FEE - side * pay
        ret = max(-1 / lev, ret)
        pnl = eq * lev * ret * risk_frac
        eq += pnl
        pnls.append(pnl)
        i = np.searchsorted(ts, t1)
    return np.array(pnls), eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lev", type=float, default=3)
    lev = ap.parse_args().lev

    print("== A) CARRY delta-neutral (spot largo + perp corto), 500 dias aprox ==")
    data = {p: load(p) for p in PAIRS}
    for p, (f, _) in data.items():
        print(f"{p}: funding medio/8h={f.rate.mean():.5%}  %positivo={(f.rate > 0).mean():.0%}  "
              f"max={f.rate.max():.4%} min={f.rate.min():.4%}")
        for thr in (0.0, 0.0001, 0.0002):
            r = carry(f, thr)
            print(f"   umbral {thr:.4%}: retorno {r['ret']:.1%} en {r['days']:.0f}d  APR {r['apr']:.1%}  "
                  f"entradas {r['trades']}")

    print(f"\n== B) DIRECCIONAL ({lev:g}x), train 60% / test 40% ==")
    rows = []
    for p, (f, px) in data.items():
        cut = int(len(f) * 0.6)
        tr, te = f.iloc[:cut], f.iloc[cut:]
        for mode, name in ((1, "contrarian"), (-1, "momentum")):
            best = None
            for thr, hold in itertools.product((0.0001, 0.0002, 0.0004), (8, 24, 72)):
                pn, eq = directional(tr, px, lev, thr, hold, mode)
                if len(pn) >= 8 and (best is None or eq > best[0]):
                    best = (eq, thr, hold)
            if not best:
                continue
            pn, eq = directional(te, px, lev, best[1], best[2], mode)
            if len(pn):
                rows.append(dict(pair=p, mode=name, thr=f"{best[1]:.4%}", hold_h=best[2],
                                 train_eq=round(best[0], 1), test_n=len(pn),
                                 test_win=f"{(pn > 0).mean():.0%}", test_eq=round(eq, 1)))
    out = pd.DataFrame(rows).sort_values("test_eq", ascending=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\nTest con equity final > 100:", int((out.test_eq > 100).sum()), "de", len(out))


if __name__ == "__main__":
    main()
