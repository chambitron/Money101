"""Descarga velas publicas de BloFin (sin API key) a CSV. Uso:
python fetch_candles.py BTC-USDT 3m 5000 data/BTC-USDT_3m.csv
"""
import sys, time, csv, requests

URL = "https://openapi.blofin.com/api/v1/market/candles"


def fetch(inst, bar, total):
    rows, after = [], None
    while len(rows) < total:
        p = {"instId": inst, "bar": bar, "limit": 100}
        if after:
            p["after"] = after
        r = requests.get(URL, params=p, timeout=20).json()["data"]
        if not r:
            break
        rows += r
        after = r[-1][0]
        time.sleep(0.2)
    rows = sorted({int(x[0]): x for x in rows}.values(), key=lambda x: int(x[0]))
    return rows


if __name__ == "__main__":
    inst, bar, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for x in fetch(inst, bar, n):
            w.writerow(x[:5] + [x[5]])
