"""Modelo de carry delta-neutral con riesgo de liquidacion (spot largo + perp corto aislado en BloFin).

Simulacion hora a hora con velas 1H y funding real de BloFin:
- Capital E repartido en n spot + n/L de margen en el perp (nocional n en ambas patas).
- El perp cobra/paga funding sobre su nocional en cada liquidacion (cada 8h).
- Liquidacion: si el maximo de la vela lleva el margen del perp a <= mmr*nocional, se pierde TODO el margen
  y la posicion corta desaparece (el spot queda sin cobertura); se vuelve a cubrir en el cierre de la vela.
- Rebalanceo por banda: si margen/nocional sale de [lo, hi] x objetivo, se reajustan ambas patas (con comisiones).
- Filtro de funding (opcional): cierra el hedge si el funding medio de las ultimas 9 liquidaciones es <= 0.
Limitaciones: usa el precio del perp para ambas patas (sin riesgo de base), sin spreads reales,
y requiere cuenta spot fuera de BloFin (BloFin solo tiene perpetuos).
Uso: python carry_model.py
"""
import numpy as np
import pandas as pd

PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
PERP_FEE, SPOT_FEE, SLIP, MMR = 0.0006, 0.001, 0.0002, 0.005
LEVS = (1, 2, 3, 5, 8)


def load(pair):
    c = pd.read_csv(f"data/long/{pair}_1H.csv").sort_values("ts").reset_index(drop=True)
    f = pd.read_csv(f"data/{pair}_funding.csv")
    rate = dict(zip(f.ts.astype("int64"), f.rate))
    c = c[c.ts >= f.ts.min()].reset_index(drop=True)
    return c, rate


def trade_cost(notional):
    return notional * (PERP_FEE + SLIP + SPOT_FEE + SLIP)


def simulate(c, rate, lev, band, filt, capital=100.0):
    ts, o, h, cl = (c[k].values for k in ("ts", "open", "high", "close"))
    recent = []
    E, q, e, m, flat = capital, 0.0, 0.0, 0.0, True
    liqs = rebal = 0
    curve, min_ratio = [], np.inf

    def open_pos(E, price):
        n = E * lev / (lev + 1)
        E -= trade_cost(n)
        n = E * lev / (lev + 1)
        return E, n / price, price, n / lev

    for i in range(len(c)):
        # funding de esta hora (precio de apertura)
        if ts[i] in rate:
            r = rate[ts[i]]
            recent = (recent + [r])[-9:]
            if not flat:
                m += r * q * o[i]  # el corto cobra si r > 0
        want = (not filt) or (len(recent) == 9 and np.mean(recent) > 0)
        if flat and want and i > 0:
            E, q, e, m = open_pos(E, cl[i])
            flat = False
        elif not flat and not want:
            w = m + q * (e - cl[i])
            E = q * cl[i] + w
            E -= trade_cost(q * cl[i])
            q = e = m = 0.0
            flat = True
        if flat:
            curve.append(E)
            continue
        wh = m + q * (e - h[i])
        if wh <= MMR * q * h[i]:  # liquidado dentro de la hora
            liqs += 1
            E = q * cl[i]  # el margen del perp se pierde por completo
            E, q, e, m = open_pos(E, cl[i])
            curve.append(E)
            continue
        w = m + q * (e - cl[i])
        ratio = w / (q * cl[i]) * lev
        min_ratio = min(min_ratio, w / (q * h[i]))
        E = q * cl[i] + w
        if band and not (band[0] <= ratio <= band[1]):
            n_old, n_new = q * cl[i], E * lev / (lev + 1)
            E -= trade_cost(abs(n_new - n_old))
            n_new = E * lev / (lev + 1)
            q, e, m = n_new / cl[i], cl[i], n_new / lev
            rebal += 1
        curve.append(E)
    curve = np.array(curve)
    dd = (1 - curve / np.maximum.accumulate(curve)).max()
    days = len(c) / 24
    return dict(final=E, ret=E / capital - 1, apr=(E / capital - 1) / days * 365, liqs=liqs, rebal=rebal,
                max_dd=dd, min_margin=min_ratio if np.isfinite(min_ratio) else np.nan)


def moves(c):
    out = {}
    for name, hrs in (("24h", 24), ("7d", 168)):
        win_low = c.low.rolling(hrs).min()
        up = (c.high.rolling(hrs).max() / win_low - 1).max()
        out[name] = up
    return out


def main():
    for pair in PAIRS:
        c, rate = load(pair)
        days = len(c) / 24
        mv = moves(c)
        print(f"\n===== {pair}  ({days:.0f} dias) =====")
        print(f"Mayor subida (min->max) en ventana: 24h {mv['24h']:.0%}, 7d {mv['7d']:.0%}")
        print(f"{'L':>2} {'politica':<22} {'neto':>7} {'APR':>7} {'liqs':>4} {'rebal':>5} {'maxDD':>6} "
              f"{'subida a liquidar':>18}")
        for lev in LEVS:
            buf = 1 / lev - MMR  # subida de precio que liquida sin rebalanceo (aprox.)
            for label, band, filt in (("sin rebalanceo", None, False),
                                      ("banda 0.6-1.6", (0.6, 1.6), False),
                                      ("banda + filtro funding", (0.6, 1.6), True)):
                r = simulate(c, rate, lev, band, filt)
                print(f"{lev:>2} {label:<22} {r['ret']:>7.1%} {r['apr']:>7.1%} {r['liqs']:>4} {r['rebal']:>5} "
                      f"{r['max_dd']:>6.1%} {buf:>17.0%}")


if __name__ == "__main__":
    main()
