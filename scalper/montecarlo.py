"""Proyecciones Montecarlo del carry delta-neutral (spot largo + perp corto) con riesgo de liquidacion.

Metodologia
- Precio: bootstrap por bloques (24 h) de velas 1H reales de BloFin (retorno, maximo y minimo relativos al cierre
  previo, remuestreados JUNTOS). Conserva colas gordas, agrupacion de volatilidad y rango intrabarra. Se centra el
  retorno medio (sin heredar la tendencia de la muestra) y se reescala la volatilidad por regimen y escenario.
- Funding: AR(1) con regresor de retorno 24h, f_t = a + phi f_{t-1} + beta r24_t + e_t, ajustado por OLS con
  residuos remuestreados (colas reales). Se simula sobre cada trayectoria de precio (funding sube tras subidas).
- Escenarios (mezcla ponderada): base ajustado por regimen, volatil (macro/geopolitica), squeeze alcista y crash,
  con saltos de Poisson. Son SUPUESTOS ajustables (SCENARIOS / --weights), no predicciones: las noticias se traducen
  en pesos y parametros de riesgo, no en un precio objetivo.
- Resultados: percentiles, P(perdida), CVaR 5%, P(>=1 liquidacion) con IC de Wilson 95%, drawdown.
Uso: python montecarlo.py [--days 30,90] [--paths 4000] [--pairs BTC-USDT,ETH-USDT,SOL-USDT]
     [--weights base=0.55,volatil=0.25,squeeze=0.10,crash=0.10] [--validate] [--goal]
"""
import argparse, math
import numpy as np
import pandas as pd

from carry_model import MMR, PERP_FEE, SLIP, trade_cost

SCENARIOS = {  # vol = multiplicador de volatilidad, drift anual, lam = saltos/dia, jump = tamano medio, sign = sesgo
    "base":    dict(vol=1.0, drift=0.0,   lam=0.00, jump=0.00, up=0.5, fshift=0.0),
    "volatil": dict(vol=1.5, drift=0.0,   lam=0.10, jump=0.04, up=0.5, fshift=0.0),
    "squeeze": dict(vol=1.3, drift=0.8,   lam=0.08, jump=0.06, up=1.0, fshift=0.00005),
    "crash":   dict(vol=1.6, drift=-0.6,  lam=0.08, jump=0.07, up=0.0, fshift=-0.0001),
}
DEFAULT_WEIGHTS = dict(base=0.55, volatil=0.25, squeeze=0.10, crash=0.10)
LEVS, BAND = (1, 2, 3, 5), (0.6, 1.6)
BLOCK = 24


# ------------------------------------------------------------------ datos y diagnosticos
def load(pair):
    c = pd.read_csv(f"data/long/{pair}_1H.csv").sort_values("ts").reset_index(drop=True)
    f = pd.read_csv(f"data/{pair}_funding.csv")
    c = c[c.ts >= f.ts.min()].reset_index(drop=True)
    prev = c.close.shift()
    h = pd.DataFrame(dict(r=np.log(c.close / prev), uh=np.log(c.high / prev), ul=np.log(c.low / prev))).dropna()
    h["r"] -= h.r.mean()
    return c, f, h.reset_index(drop=True)


def hill(x, frac=0.05):
    x = np.sort(np.abs(x))[::-1]
    k = max(10, int(len(x) * frac))
    return 1 / np.mean(np.log(x[:k] / x[k]))


def diagnostics(pair, c, h):
    r = h.r.values
    z = (r - r.mean()) / r.std()
    a = np.abs(r)
    acf = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1]
    print(f"  Diagnostico {pair}: vol anual {r.std() * math.sqrt(8760):.0%} | curtosis exceso {np.mean(z ** 4) - 3:.1f} | "
          f"asimetria {np.mean(z ** 3):+.2f} | indice de cola (Hill) {hill(r):.1f} | "
          f"ACF |r| lag1/24: {acf(a, 1):.2f}/{acf(a, 24):.2f}  -> colas gordas y volatilidad agrupada: se usa bootstrap por bloques")


def fit_funding(c, f):
    ts = c.ts.values
    close = c.set_index("ts").close
    rows = []
    fl = f.reset_index(drop=True)
    for i in range(1, len(fl)):
        t = int(fl.ts[i])
        if t in close.index and (t - 24 * 3_600_000) in close.index:
            rows.append((fl.rate[i], fl.rate[i - 1], np.log(close[t] / close[t - 24 * 3_600_000])))
    d = np.array(rows)
    X = np.column_stack([np.ones(len(d)), d[:, 1], d[:, 2]])
    beta, *_ = np.linalg.lstsq(X, d[:, 0], rcond=None)
    res = d[:, 0] - X @ beta
    r2 = 1 - res.var() / d[:, 0].var()
    return dict(a=beta[0], phi=beta[1], b=beta[2], res=res, r2=r2, f0=float(fl.rate.iloc[-1]),
                mu=beta[0] / (1 - beta[1]), half=math.log(0.5) / math.log(beta[1]) * 8 / 24)


def context(pair, c, f, h):
    cl, n = c.close.values, len(c)
    ret = lambda d: cl[-1] / cl[-24 * d] - 1
    vol = lambda d: np.std(np.diff(np.log(cl[-24 * d:]))) * math.sqrt(8760)
    full = h.r.std() * math.sqrt(8760)
    ema = pd.Series(cl).ewm(span=24 * 50, adjust=False).mean().iloc[-1]
    roll = pd.Series(np.log(cl)).diff().rolling(24 * 30).std().dropna().values * math.sqrt(8760)
    pct = (roll < vol(30)).mean()
    fr = f.rate.values
    fp = (fr < fr[-21:].mean()).mean()
    dd = cl[-1] / cl[-24 * 90:].max() - 1
    print(f"  Contexto {pair}: precio {cl[-1]:,.2f} | 7d {ret(7):+.1%} | 30d {ret(30):+.1%} | 90d {ret(90):+.1%} | "
          f"vs EMA50d {cl[-1] / ema - 1:+.1%} | caida desde max 90d {dd:.1%}")
    print(f"      vol 30d {vol(30):.0%} (percentil {pct:.0%} de su historia; vol total {full:.0%}) | "
          f"funding 7d {fr[-21:].mean():.4%}/8h (percentil {fp:.0%} de su historia)")
    return float(np.clip(0.5 + 0.5 * vol(30) / full, 0.6, 1.6))  # multiplicador de regimen (50% encogido a 1)


# ------------------------------------------------------------------ generacion de trayectorias
def gen_paths(h, fm, n, H, sc, k0, rng):
    T = len(h)
    nb = math.ceil(H / BLOCK)
    idx = ((rng.integers(0, T, (n, nb))[:, :, None] + np.arange(BLOCK)) % T).reshape(n, -1)[:, :H]
    k = k0 * sc["vol"]
    r, uh = h.r.values[idx] * k, h.uh.values[idx] * k
    r = r + sc["drift"] / 8760
    uh = uh + sc["drift"] / 8760
    if sc["lam"] > 0:
        jump = rng.random((n, H)) < sc["lam"] / 24
        sgn = np.where(rng.random((n, H)) < sc["up"], 1.0, -1.0)
        size = sgn * sc["jump"] * np.exp(0.35 * rng.standard_normal((n, H)) - 0.06)
        r = r + jump * size
    uh = np.maximum(uh, r)
    ul = np.minimum(h.ul.values[idx] * k, r)
    # funding en cada liquidacion (cada 8 h): AR(1) + retorno 24h simulado + residuo remuestreado
    cum = np.cumsum(r, axis=1)
    F = np.zeros((n, H), dtype=np.float64)
    f = np.full(n, fm["f0"])
    for t in range(0, H, 8):
        r24 = cum[:, t] - (cum[:, t - 24] if t >= 24 else 0.0)
        f = fm["a"] + fm["phi"] * f + fm["b"] * r24 + rng.choice(fm["res"], n) + sc["fshift"]
        f = np.clip(f, -0.004, 0.004)
        F[:, t] = f
    return r, uh, ul, F


def simulate_carry(r, uh, F, lev, rebalance, capital=100.0):
    n, H = r.shape
    E0 = capital - trade_cost(capital * lev / (lev + 1))
    nn = E0 * lev / (lev + 1)
    prev = np.ones(n)
    q, e, m = np.full(n, nn), np.ones(n), np.full(n, nn / lev)  # precio inicial normalizado a 1 -> q = n
    liqs, rebs = np.zeros(n, int), np.zeros(n, int)
    peak, mdd = np.full(n, capital), np.zeros(n)
    for t in range(H):
        o = prev
        c, hi = o * np.exp(r[:, t]), o * np.exp(uh[:, t])
        m = m + F[:, t] * q * o
        liq = (m + q * (e - hi)) <= MMR * q * hi
        w = m + q * (e - c)
        ratio = w / (q * c) * lev
        reb = (~liq) & rebalance & ((ratio < BAND[0]) | (ratio > BAND[1]))
        act = liq | reb
        if act.any():
            E = np.where(liq, q * c, q * c + w)  # liquidado: se pierde el margen del perp
            n_old = q * c
            n_new = E * lev / (lev + 1)
            E = E - trade_cost(np.where(liq, n_new, np.abs(n_new - n_old)))
            n_new = E * lev / (lev + 1)
            q = np.where(act, n_new / c, q)
            e = np.where(act, c, e)
            m = np.where(act, n_new / lev, m)
            liqs += liq
            rebs += reb
        prev = c
        eq = q * c + m + q * (e - c)
        peak = np.maximum(peak, eq)
        mdd = np.maximum(mdd, 1 - eq / peak)
    return eq, liqs, rebs, mdd


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - s), min(1, c + s)


# ------------------------------------------------------------------ reportes
def run_pair(pair, days, n_paths, weights, rng, rows):
    c, f, h = load(pair)
    print(f"\n=================== {pair} ===================")
    diagnostics(pair, c, h)
    k0 = context(pair, c, f, h)
    fm = fit_funding(c, f)
    print(f"  Funding AR(1): media larga {fm['mu']:.4%}/8h | phi {fm['phi']:.2f} (vida media {fm['half']:.1f} dias) | "
          f"beta retorno24h {fm['b']:+.4f} | R2 {fm['r2']:.2f} | actual {fm['f0']:.4%}")
    print(f"  Multiplicador de volatilidad por regimen (vol reciente vs historica): x{k0:.2f}")
    for d in days:
        H = d * 24
        parts = {s: [] for s in weights}
        counts = {s: int(round(n_paths * w)) for s, w in weights.items()}
        paths = {s: gen_paths(h, fm, counts[s], H, SCENARIOS[s], k0, rng) for s in weights if counts[s] > 0}
        print(f"\n  --- Horizonte {d} dias ({n_paths} trayectorias; pesos "
              + ", ".join(f"{s} {w:.0%}" for s, w in weights.items()) + ") ---")
        print(f"  {'L':>2} {'politica':<15} {'esc.':<8} {'retorno med':>11} {'mediana':>8} {'P5':>7} {'P95':>7} "
              f"{'P(perd)':>8} {'CVaR5':>7} {'P(liq) [IC95]':>20} {'reb':>5} {'DD p95':>7}")
        for lev in LEVS:
            for pol, reb in (("sin rebalanceo", False), ("banda 0.6-1.6", True)):
                allr, alll, allm, allb = [], [], [], []
                for s, (r, uh, ul, F) in paths.items():
                    eq, lq, rb, mdd = simulate_carry(r, uh, F, lev, reb)
                    ret = eq / 100 - 1
                    allr.append(ret), alll.append(lq), allm.append(mdd), allb.append(rb)
                    if s != "base" or len(paths) == 1 or True:
                        rows.append(dict(pair=pair, days=d, lev=lev, policy=pol, scenario=s,
                                         mean=ret.mean(), p5=np.percentile(ret, 5), p95=np.percentile(ret, 95),
                                         p_loss=(ret < 0).mean(), p_liq=(lq > 0).mean()))
                ret, lq, mdd, rb = map(np.concatenate, (allr, alll, allm, allb))
                k, N = int((lq > 0).sum()), len(ret)
                lo, hi = wilson(k, N)
                cv = ret[ret <= np.percentile(ret, 5)].mean()
                rows.append(dict(pair=pair, days=d, lev=lev, policy=pol, scenario="MEZCLA", mean=ret.mean(),
                                 p5=np.percentile(ret, 5), p95=np.percentile(ret, 95), p_loss=(ret < 0).mean(),
                                 p_liq=k / N, cvar5=cv, p_liq_lo=lo, p_liq_hi=hi, dd95=np.percentile(mdd, 95)))
                print(f"  {lev:>2} {pol:<15} {'MEZCLA':<8} {ret.mean():>11.2%} {np.median(ret):>8.2%} "
                      f"{np.percentile(ret, 5):>7.2%} {np.percentile(ret, 95):>7.2%} {(ret < 0).mean():>8.1%} "
                      f"{cv:>7.2%} {k / N:>7.1%} [{lo:.1%},{hi:.1%}] {rb.mean():>5.1f} {np.percentile(mdd, 95):>7.2%}")
        # detalle por escenario para L=3 con rebalanceo
        print(f"  Detalle por escenario (L=3, banda), {d} dias:")
        for s, (r, uh, ul, F) in paths.items():
            eq, lq, rb, mdd = simulate_carry(r, uh, F, 3, True)
            ret = eq / 100 - 1
            eq2, lq2, _, _ = simulate_carry(r, uh, F, 3, False)
            print(f"     {s:<8} retorno med {ret.mean():>7.2%}  P5 {np.percentile(ret, 5):>7.2%}  "
                  f"P(perd) {(ret < 0).mean():>5.1%}  P(liq) con banda {(lq > 0).mean():>5.1%} / sin banda {(lq2 > 0).mean():>5.1%}")


def validate(pair, rng):
    from carry_model import simulate as hist_sim, load as hload
    c, f, h = load(pair)
    fm = fit_funding(c, f)
    H = len(c) - 1
    r, uh, ul, F = gen_paths(h, fm, 1500, H, SCENARIOS["base"], 1.0, rng)
    eq, *_ = simulate_carry(r, uh, F, 3, True)
    ret = eq / 100 - 1
    cc, rate = hload(pair)
    real = hist_sim(cc, rate, 3, (0.6, 1.6), False)["ret"]
    print(f"\nValidacion {pair}: retorno real a {H / 24:.0f} dias (L=3, banda) = {real:.1%} vs Montecarlo base "
          f"P5 {np.percentile(ret, 5):.1%} / mediana {np.median(ret):.1%} / P95 {np.percentile(ret, 95):.1%} "
          f"-> percentil del real: {(ret < real).mean():.0%}")


def goal(weights, rng, h, fm):
    print("\n=== Viabilidad de la meta original (100 -> 1000 USD en 24 h, un solo trade apalancado) ===")
    print("Probabilidad de que UNA posicion todo-adentro llegue a 10x ANTES de liquidarse, trayectorias Montecarlo 24h:")
    print(f"  {'L':>3} {'largo':>9} {'corto':>9}")
    n = 40000
    parts = {s: gen_paths(h, fm, int(round(n * w)), 24, SCENARIOS[s], 1.0, rng) for s, w in weights.items() if w > 0}
    r = np.concatenate([p[0] for p in parts.values()])
    uh = np.concatenate([p[1] for p in parts.values()])
    ul = np.concatenate([p[2] for p in parts.values()])
    prev = np.cumsum(r, axis=1) - r
    hi, lo = np.exp(prev + uh), np.exp(prev + ul)
    for L in (3, 5, 10, 20, 50):
        fee = L * 2 * (PERP_FEE + SLIP)
        res = []
        for side in (1, -1):
            tgt, liq = (1 + (9 + fee) / L, 1 - (1 / L - MMR)) if side == 1 else (1 - (9 + fee) / L, 1 + (1 / L - MMR))
            if side == -1 and tgt <= 0:
                res.append(0.0)
                continue
            t_hit = np.where((hi >= tgt).any(1), (hi >= tgt).argmax(1), 99) if side == 1 else \
                np.where((lo <= tgt).any(1), (lo <= tgt).argmax(1), 99)
            l_hit = np.where((lo <= liq).any(1), (lo <= liq).argmax(1), 99) if side == 1 else \
                np.where((hi >= liq).any(1), (hi >= liq).argmax(1), 99)
            res.append(np.mean((t_hit < 99) & (t_hit < l_hit)))
        print(f"  {L:>3}x {res[0]:>9.3%} {res[1]:>9.3%}")
    print("  (un trade 10x requiere mover el precio 9/L; con L alto la liquidacion llega mucho antes que la meta)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="BTC-USDT,ETH-USDT,SOL-USDT")
    ap.add_argument("--days", default="30,90")
    ap.add_argument("--paths", type=int, default=4000)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--goal", action="store_true")
    a = ap.parse_args()
    w = dict(DEFAULT_WEIGHTS)
    if a.weights:
        w = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    rng = np.random.default_rng(a.seed)
    rows = []
    pairs = a.pairs.split(",")
    for p in pairs:
        run_pair(p, [int(x) for x in a.days.split(",")], a.paths, w, rng, rows)
    pd.DataFrame(rows).to_csv("montecarlo_results.csv", index=False)
    if a.validate:
        for p in pairs:
            validate(p, rng)
    if a.goal:
        c, f, h = load(pairs[0])
        goal(w, rng, h, fit_funding(c, f))
    print("\nResultados completos en montecarlo_results.csv")


if __name__ == "__main__":
    main()
