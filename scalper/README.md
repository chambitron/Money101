# scalper — investigacion y paper trading en BloFin (solo datos publicos)

Nada aqui opera con dinero real ni usa API keys. Todo es backtest, Montecarlo o simulacion en papel.

| Script | Para que |
|---|---|
| `fetch_candles.py`, `fetch_funding.py` | Descargan velas y funding publicos a `data/` (`data/long/` para 1H largo). |
| `backtest.py`, `research.py`, `tv_test.py` | Backtests de scalping 3m-1H (indicadores clasicos y estrategias publicas de TradingView). Ninguna dio ventaja fuera de muestra. |
| `funding_research.py`, `carry_model.py` | Carry delta-neutral (spot largo + perp corto): retorno historico, liquidacion, rebalanceo. |
| `paper_carry.py` | Monitor de paper trading del carry, ciclo de 3 min, alertas de margen, resumen diario (`--report`). |
| `montecarlo.py` | Proyecciones Montecarlo del carry (bootstrap por bloques + funding AR(1) + escenarios). |

Montecarlo: `python montecarlo.py --days 30,90 --paths 4000 --validate --goal`
Escenarios y pesos: `SCENARIOS` en `montecarlo.py` o `--weights base=0.55,volatil=0.25,squeeze=0.10,crash=0.10`.
Alertas por webhook (opcional): `export ALERT_WEBHOOK_URL=...` antes de correr `paper_carry.py`.

Limitaciones: precio del perp para ambas patas (sin riesgo de base), rebalanceo ideal cada hora, sin spreads
reales, y el spot requiere un exchange aparte (BloFin solo tiene perpetuos). Los escenarios son supuestos, no predicciones.
