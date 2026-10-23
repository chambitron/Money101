"""Descarga el historial publico de funding rate de BloFin. Uso:
python fetch_funding.py BTC-USDT 1000 data/BTC-USDT_funding.csv
"""
import sys, time, csv, requests

URL = "https://openapi.blofin.com/api/v1/market/funding-rate-history"


def fetch(inst, total):
    rows, before = {}, None
    while len(rows) < total:
        p = {"instId": inst, "limit": 100}
        if before:
            p["after"] = before
        for attempt in range(5):
            try:
                r = requests.get(URL, params=p, timeout=20).json()["data"]
                break
            except (requests.RequestException, KeyError, ValueError):
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("BloFin no responde")
        if not r:
            break
        for x in r:
            rows[int(x["fundingTime"])] = float(x["fundingRate"])
        before = min(int(x["fundingTime"]) for x in r)
        time.sleep(0.2)
    return sorted(rows.items())


if __name__ == "__main__":
    inst, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "rate"])
        w.writerows(fetch(inst, n))
