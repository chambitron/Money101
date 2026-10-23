"""Monitor de PAPER TRADING del carry delta-neutral (spot largo + perp corto) con ciclo de 3 minutos.

Solo lee datos PUBLICOS de BloFin. No usa API keys, no coloca ordenes y no toca ninguna cuenta:
todo es simulado y se guarda en paper_state.json (estado) y paper_log.csv (un registro por ciclo y par).

Cada ciclo:
  1. Lee precio de marca, velas de 3m confirmadas (maximos/minimos desde el ciclo anterior) y funding.
  2. Revisa liquidacion del perp simulado con el maximo del periodo (se pierde todo el margen).
  3. Si paso la hora de liquidacion de funding (00/08/16 UTC), aplica el funding real (historial de BloFin).
  4. Rebalancea si margen/objetivo sale de la banda (con comisiones).
Uso:
  python paper_carry.py                        # ciclo continuo cada 180 s (BTC, ETH, SOL; 100 USD; 3x)
  python paper_carry.py --once                 # un solo ciclo (prueba)
  python paper_carry.py --lev 2 --capital 100 --pairs BTC-USDT,ETH-USDT
  python paper_carry.py --reset                # borra el estado y empieza de cero
"""
import argparse, csv, json, os, time
import requests

from carry_model import MMR, trade_cost

BASE = "https://openapi.blofin.com/api/v1/market"
STATE, LOG = "paper_state.json", "paper_log.csv"
BAND = (0.6, 1.6)
DEFAULT_PAIRS = "BTC-USDT,ETH-USDT,SOL-USDT"


def get(path, **params):
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=15).json()
            if r.get("code") == "0":
                return r["data"]
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"BloFin no responde: {path}")


def open_position(E, price, lev):
    n = E * lev / (lev + 1)
    E -= trade_cost(n)
    n = E * lev / (lev + 1)
    return dict(q=n / price, e=price, m=n / lev)


def new_state(capital, price, lev, now):
    st = dict(capital=capital, lev=lev, started=now, last_ts=now, funding_net=0.0, fees=0.0,
              liqs=0, rebal=0, min_buffer=1.0, pending=None, **open_position(capital, price, lev))
    st["fees"] = capital - (st["q"] * price + st["m"])
    return st


def equity(st, price):
    return st["q"] * price + st["m"] + st["q"] * (st["e"] - price)


def step(st, price, high, low, now, settled_rates=()):
    """Avanza un ciclo con el estado, el precio de marca, max/min del periodo y los funding ya liquidados.
    settled_rates: lista de tasas de funding liquidadas desde el ciclo anterior. Devuelve (estado, eventos)."""
    events, lev = [], st["lev"]
    for r in settled_rates:
        pay = r * st["q"] * price  # el corto cobra si r > 0
        st["m"] += pay
        st["funding_net"] += pay
        events.append(f"funding {r:+.5%} -> {pay:+.4f} USDT")
    wh = st["m"] + st["q"] * (st["e"] - high)
    st["min_buffer"] = min(st["min_buffer"], wh / (st["q"] * high))
    if wh <= MMR * st["q"] * high:
        st["liqs"] += 1
        E = st["q"] * price  # el margen del perp se pierde; el spot queda
        old = E
        st.update(open_position(E, price, lev))
        st["fees"] += old - (st["q"] * price + st["m"])
        events.append("LIQUIDADO: margen perdido, se re-cubre")
    else:
        w = st["m"] + st["q"] * (st["e"] - price)
        ratio = w / (st["q"] * price) * lev
        if not BAND[0] <= ratio <= BAND[1]:
            E = st["q"] * price + w
            n_old, n_new = st["q"] * price, E * lev / (lev + 1)
            cost = trade_cost(abs(n_new - n_old))
            E -= cost
            n_new = E * lev / (lev + 1)
            st.update(q=n_new / price, e=price, m=n_new / lev)
            st["fees"] += cost
            st["rebal"] += 1
            events.append(f"REBALANCEO (ratio {ratio:.2f}x objetivo, costo {cost:.4f})")
    st["last_ts"] = now
    return st, events


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, STATE)


def cycle(states, pairs, capital, lev):
    now = int(time.time() * 1000)
    new_log = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as lf:
        w = csv.writer(lf)
        if new_log:
            w.writerow(["ts", "pair", "price", "high", "low", "equity", "pnl", "funding_net", "fees",
                        "margin_ratio", "pending_rate", "liqs", "rebal", "events"])
        for p in pairs:
            price = float(get("mark-price", instId=p)[0]["markPrice"])
            candles = [c for c in get("candles", instId=p, bar="3m", limit=5) if c[8] == "1"]
            if p not in states:
                states[p] = new_state(capital, price, lev, now)
                print(f"[{p}] posicion simulada abierta @ {price} (capital {capital} USDT, {lev:g}x)")
            st = states[p]
            since = [c for c in candles if int(c[0]) + 180_000 > st["last_ts"]]
            high = max([price] + [float(c[2]) for c in since])
            low = min([price] + [float(c[3]) for c in since])
            fr = get("funding-rate", instId=p)[0]
            settled = []
            pend = st.get("pending")
            if pend and now >= pend["time"] and pend["time"] > st.get("applied", 0):
                try:  # tasa exacta liquidada; si falla, la ultima tasa vista antes del cobro
                    hist = get("funding-rate-history", instId=p, limit=3)
                    exact = {int(h["fundingTime"]): float(h["fundingRate"]) for h in hist}
                    settled.append(exact.get(pend["time"], pend["rate"]))
                except RuntimeError:
                    settled.append(pend["rate"])
                st["applied"] = pend["time"]
            st["pending"] = dict(time=int(fr["fundingTime"]), rate=float(fr["fundingRate"]))
            st, events = step(st, price, high, low, now, settled)
            E = equity(st, price)
            ratio = (st["m"] + st["q"] * (st["e"] - price)) / (st["q"] * price) * st["lev"]
            w.writerow([now, p, price, high, low, round(E, 4), round(E - st["capital"], 4),
                        round(st["funding_net"], 4), round(st["fees"], 4), round(ratio, 3),
                        st["pending"]["rate"], st["liqs"], st["rebal"], "; ".join(events)])
            days = (now - st["started"]) / 86_400_000
            apr = f"{(E / st['capital'] - 1) / days * 365:.1%}" if days >= 1 else "n/d (<1 dia)"
            print(f"{time.strftime('%H:%M:%S')} {p:<9} {price:>10.2f}  equity {E:8.4f}  pnl {E - st['capital']:+.4f}  "
                  f"funding {st['funding_net']:+.4f}  ratio {ratio:.2f}  next {st['pending']['rate']:+.4%}  "
                  f"liq {st['liqs']} reb {st['rebal']}  APR~{apr}" + (f"  !! {'; '.join(events)}" if events else ""))
    save_state(states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--lev", type=float, default=3.0)
    ap.add_argument("--interval", type=int, default=180, help="segundos entre ciclos (3 min = 180)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        for f in (STATE, LOG):
            if os.path.exists(f):
                os.remove(f)
    states, pairs = load_state(), a.pairs.split(",")
    while True:
        t0 = time.time()
        try:
            cycle(states, pairs, a.capital, a.lev)
        except Exception as ex:  # un fallo de red no debe matar el monitor
            print(f"{time.strftime('%H:%M:%S')} error en el ciclo: {ex}")
        if a.once:
            break
        time.sleep(max(1, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
