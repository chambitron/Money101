"""Grafico del PnL del paper trading del carry (HTML autocontenido, sin dependencias).

Lee paper_log.csv y escribe pnl_chart.html con:
  - PnL acumulado en el tiempo por par (linea),
  - PnL por dia UTC por par (barras agrupadas),
  - tarjetas resumen y una tabla diaria (vista alternativa a los graficos).
Un solo eje por grafico. Colores fijos por par (BTC azul, ETH naranja, SOL aqua), validados en claro y oscuro.
Uso: python pnl_chart.py [paper_log.csv] [pnl_chart.html]   (paper_carry.py lo regenera en cada ciclo)
"""
import csv, datetime, json, sys
from collections import OrderedDict

LOG, OUT = "paper_log.csv", "pnl_chart.html"
PAIR_SLOT = {"BTC-USDT": 1, "ETH-USDT": 2, "SOL-USDT": 3}  # el color sigue a la entidad, no a su orden


def _utc(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc)


def load(log=LOG):
    series = OrderedDict()
    with open(log) as f:
        for r in csv.DictReader(f):
            series.setdefault(r["pair"], []).append(dict(
                ts=int(r["ts"]), pnl=float(r["pnl"]), fund=float(r["funding_net"]), fees=float(r["fees"]),
                liqs=int(r["liqs"]), reb=int(r["rebal"])))
    return series


def prepare(series):
    pairs, days = [], OrderedDict()
    for i, (pair, rows) in enumerate(series.items()):
        hourly = OrderedDict()
        for r in rows:  # ultima lectura de cada hora (el HTML no necesita un punto cada 3 min)
            hourly[r["ts"] // 3_600_000] = [r["ts"], round(r["pnl"], 4)]
        line = list(hourly.values())
        if line[-1][0] != rows[-1]["ts"]:
            line.append([rows[-1]["ts"], round(rows[-1]["pnl"], 4)])
        last_of_day = OrderedDict()
        for r in rows:
            last_of_day[_utc(r["ts"]).strftime("%Y-%m-%d")] = r
        prev = dict(pnl=0.0, fund=0.0, fees=0.0)
        for d, r in last_of_day.items():
            days.setdefault(d, {})[pair] = dict(
                pnl=round(r["pnl"] - prev["pnl"], 4), fund=round(r["fund"] - prev["fund"], 4),
                fees=round(r["fees"] - prev["fees"], 4), liqs=r["liqs"], reb=r["reb"])
            prev = r
        pairs.append(dict(id=pair, slot=PAIR_SLOT.get(pair, 4 + i), line=line, last=rows[-1]))
    today = _utc(max(r["ts"] for rows in series.values() for r in rows)).strftime("%Y-%m-%d")
    for p in pairs:
        p["last"] = dict(pnl=round(p["last"]["pnl"], 4), fund=round(p["last"]["fund"], 4),
                         liqs=p["last"]["liqs"], reb=p["last"]["reb"])
    return dict(pairs=pairs, days=[dict(date=d, v=v) for d, v in days.items()], today=today,
                updated=_utc(max(p["line"][-1][0] for p in pairs)).strftime("%Y-%m-%d %H:%M UTC"))


def build(log=LOG, out=OUT):
    series = load(log)
    if not series:
        raise SystemExit("paper_log.csv vacio: corre paper_carry.py primero")
    html = TEMPLATE.replace("__DATA__", json.dumps(prepare(series), separators=(",", ":")))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="180">
<title>PnL del carry</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--panel:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#74736d;--grid:#e8e7e3;--axis:#c9c8c2;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--good:#0ca30c;--crit:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#1a1a19;--panel:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--ink3:#9a998f;--grid:#2f2f2c;--axis:#454542;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--panel:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--ink3:#9a998f;--grid:#2f2f2c;--axis:#454542;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 40px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:28px 0 4px}.sub{color:var(--ink2);margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.tile{border:1px solid var(--grid);border-radius:8px;padding:10px 12px}.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.tile span{color:var(--ink2);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:6px 0 8px;color:var(--ink2);font-size:13px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.chart{position:relative}svg{display:block;width:100%;height:auto;overflow:visible}
.tt{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--axis);border-radius:6px;padding:6px 8px;
font-size:12px;line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,.18);display:none;white-space:nowrap;z-index:2;font-variant-numeric:tabular-nums}
.tt .h{color:var(--ink2);margin-bottom:2px}.tt i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}
details{margin-top:20px}summary{cursor:pointer;color:var(--ink2)}
table{border-collapse:collapse;width:100%;margin-top:8px;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--ink2);font-weight:600}.note{color:var(--ink3);font-size:12px;margin-top:14px}
text{fill:var(--ink2);font-size:11px}.lab{fill:var(--ink);font-size:12px;font-weight:600}
</style></head><body><main>
<h1>PnL del carry — paper trading</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<div class="legend" id="legend"></div>
<h2>PnL acumulado (USDT)</h2>
<div class="chart" id="c1"></div>
<h2>PnL por día UTC (USDT)</h2>
<div class="chart" id="c2"></div>
<details><summary>Ver tabla diaria</summary><div id="tbl"></div></details>
<p class="note">Simulación con datos públicos de BloFin: no hay órdenes reales. El PnL incluye comisiones de entrada y de rebalanceo; el día en curso es parcial.</p>
</main>
<script id="d" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('d').textContent);
const C=s=>`var(--s${s})`, f2=(v,s)=>(v=Math.abs(v)<1e-9?0:v,s&&v>0?'+':'')+v.toLocaleString('es',{minimumFractionDigits:2,maximumFractionDigits:2}),
f4=v=>(v>0?'+':'')+v.toLocaleString('es',{minimumFractionDigits:4,maximumFractionDigits:4});
const NS='http://www.w3.org/2000/svg';
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function ticks(lo,hi,n){if(lo===hi){lo-=1;hi+=1}const raw=(hi-lo)/n,m=Math.pow(10,Math.floor(Math.log10(raw))),
 st=[1,2,2.5,5,10].map(x=>x*m).find(x=>x>=raw*.95),a=Math.floor(lo/st)*st,r=[];for(let v=a;v<=hi+st*.5;v+=st)r.push(+v.toFixed(10));return r}
const tot=D.pairs.reduce((a,p)=>a+p.last.pnl,0), fund=D.pairs.reduce((a,p)=>a+p.last.fund,0),
liq=D.pairs.reduce((a,p)=>a+p.last.liqs,0), reb=D.pairs.reduce((a,p)=>a+p.last.reb,0),
td=D.days.find(d=>d.date===D.today), today=td?Object.values(td.v).reduce((a,v)=>a+v.pnl,0):0;
document.getElementById('sub').textContent=`Actualizado ${D.updated} · ${D.pairs.length} pares · ${D.days.length} día(s) de datos`;
document.getElementById('tiles').innerHTML=[['PnL total',f4(tot)+' USDT'],['PnL hoy (parcial)',f4(today)+' USDT'],
 ['Funding acumulado',f4(fund)+' USDT'],['Liquidaciones / rebalanceos',liq+' / '+reb]].map(([a,b])=>`<div class="tile"><span>${a}</span><b>${b}</b></div>`).join('');
document.getElementById('legend').innerHTML=D.pairs.map(p=>`<span><i style="background:${C(p.slot)}"></i>${p.id}</span>`).join('');
const tt=document.createElement('div');tt.className='tt';document.body.appendChild(tt);
function showTT(ev,html){tt.innerHTML=html;tt.style.display='block';const w=tt.offsetWidth,x=Math.min(ev.pageX+14,scrollX+innerWidth-w-8);tt.style.left=x+'px';tt.style.top=(ev.pageY+14)+'px'}
const hideTT=()=>tt.style.display='none';
function frame(host,H,lo,hi,padR){host.innerHTML='';const W=Math.max(320,host.clientWidth),p={l:56,r:padR,t:10,b:26},
 s=el('svg',{viewBox:`0 0 ${W} ${H}`},host),y=v=>p.t+(1-(v-lo)/(hi-lo))*(H-p.t-p.b);
 for(const v of ticks(lo,hi,5)){if(v<lo-1e-9||v>hi+1e-9)continue;el('line',{x1:p.l,x2:W-p.r,y1:y(v),y2:y(v),stroke:v===0?'var(--axis)':'var(--grid)','stroke-width':v===0?1.5:1},s);
  const t=el('text',{x:p.l-8,y:y(v)+4,'text-anchor':'end'},s);t.textContent=f2(v)}
 return{W,H,p,s,y}}
function drawLine(){const host=document.getElementById('c1'),all=D.pairs.flatMap(p=>p.line);
 let lo=Math.min(0,...all.map(a=>a[1])),hi=Math.max(0,...all.map(a=>a[1])),pad=(hi-lo)*.12||.05;lo-=pad;hi+=pad;
 const t0=Math.min(...all.map(a=>a[0])),t1=Math.max(...all.map(a=>a[0])),F=frame(host,300,lo,hi,96),{W,H,p,s,y}=F,
 x=t=>p.l+(t-t0)/Math.max(1,t1-t0)*(W-p.l-p.r),span=t1-t0,k=span>2*864e5;
 const xt=[];if(k){const nd=Math.ceil(span/864e5),stp=Math.ceil(nd/6)*864e5;for(let t=Math.ceil(t0/864e5)*864e5;t<=t1;t+=stp)xt.push(t)}
 else for(let i=0;i<=4;i++)xt.push(t0+span*i/4);
 xt.forEach((t,i)=>{const d=new Date(t),lab=k?d.toLocaleDateString('es',{day:'2-digit',month:'2-digit',timeZone:'UTC'}):d.toLocaleTimeString('es',{hour:'2-digit',minute:'2-digit',timeZone:'UTC'});
  const e=el('text',{x:x(t),y:H-6,'text-anchor':!k&&i===0?'start':!k&&i===4?'end':'middle'},s);e.textContent=lab});
 const lab=[];D.pairs.forEach(pr=>{const pts=pr.line,d=pts.map((a,i)=>(i?'L':'M')+x(a[0]).toFixed(1)+' '+y(a[1]).toFixed(1)).join(' ');
  if(pts.length===1)el('circle',{cx:x(pts[0][0]),cy:y(pts[0][1]),r:4,fill:C(pr.slot)},s);else el('path',{d,fill:'none',stroke:C(pr.slot),'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'},s);
  const e=pts[pts.length-1];el('circle',{cx:x(e[0]),cy:y(e[1]),r:4,fill:C(pr.slot),stroke:'var(--bg)','stroke-width':2},s);lab.push([y(e[1]),pr,e[1]])});
 lab.sort((a,b)=>a[0]-b[0]);for(let i=1;i<lab.length;i++)if(lab[i][0]-lab[i-1][0]<14)lab[i][0]=lab[i-1][0]+14;
 lab.forEach(([yy,pr,v])=>{const e=el('text',{x:W-p.r+10,y:yy+4,class:'lab'},s);e.textContent=pr.id.split('-')[0]+' '+f2(v,1)});
 const cross=el('line',{y1:p.t,y2:H-p.b,stroke:'var(--ink3)','stroke-width':1,visibility:'hidden'},s),dots=D.pairs.map(pr=>el('circle',{r:4,fill:C(pr.slot),stroke:'var(--bg)','stroke-width':2,visibility:'hidden'},s)),
 hit=el('rect',{x:p.l,y:p.t,width:W-p.l-p.r,height:H-p.t-p.b,fill:'transparent'},s);
 hit.addEventListener('mousemove',ev=>{const r=s.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width,t=t0+(px-p.l)/(W-p.l-p.r)*(t1-t0);
  let rows='',cx=0;D.pairs.forEach((pr,i)=>{let b=pr.line[0];for(const a of pr.line)if(Math.abs(a[0]-t)<Math.abs(b[0]-t))b=a;
   cx=x(b[0]);dots[i].setAttribute('cx',cx);dots[i].setAttribute('cy',y(b[1]));dots[i].setAttribute('visibility','visible');
   rows+=`<div><i style="background:${C(pr.slot)}"></i>${pr.id} <b>${f4(b[1])}</b></div>`});
  cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.setAttribute('visibility','visible');
  showTT(ev,`<div class="h">${new Date(t).toLocaleString('es',{timeZone:'UTC',dateStyle:'short',timeStyle:'short'})} UTC</div>${rows}`)});
 hit.addEventListener('mouseleave',()=>{cross.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'));hideTT()})}
function drawBars(){const host=document.getElementById('c2'),vals=D.days.flatMap(d=>Object.values(d.v).map(v=>v.pnl));
 let lo=Math.min(0,...vals),hi=Math.max(0,...vals),pad=(hi-lo)*.12||.05;lo-=pad;hi+=pad;
 const F=frame(host,260,lo,hi,16),{W,H,p,s,y}=F,n=D.days.length,gw=(W-p.l-p.r)/n,np=D.pairs.length,bw=Math.min(36,(gw*.7)/np-2);
 D.days.forEach((d,i)=>{const gx=p.l+gw*i+gw/2,tx=el('text',{x:gx,y:H-6,'text-anchor':'middle'},s);tx.textContent=d.date.slice(5).split('-').reverse().join('/')+(d.date===D.today?'*':'');
  D.pairs.forEach((pr,j)=>{const v=(d.v[pr.id]||{pnl:0}).pnl,bx=gx-(np*(bw+2)-2)/2+j*(bw+2),y0=y(0),y1=y(v),top=Math.min(y0,y1),h=Math.max(1,Math.abs(y1-y0)),r=Math.min(4,h);
   const up=v>=0,pth=up?`M${bx} ${y0}V${top+r}Q${bx} ${top} ${bx+r} ${top}H${bx+bw-r}Q${bx+bw} ${top} ${bx+bw} ${top+r}V${y0}Z`:
    `M${bx} ${y0}V${top+h-r}Q${bx} ${top+h} ${bx+r} ${top+h}H${bx+bw-r}Q${bx+bw} ${top+h} ${bx+bw} ${top+h-r}V${y0}Z`;
   el('path',{d:pth,fill:C(pr.slot)},s)});
  const hit=el('rect',{x:p.l+gw*i,y:p.t,width:gw,height:H-p.t-p.b,fill:'transparent'},s);
  hit.addEventListener('mousemove',ev=>showTT(ev,`<div class="h">${d.date}${d.date===D.today?' (parcial)':''}</div>`+D.pairs.map(pr=>{const v=d.v[pr.id];
   return v?`<div><i style="background:${C(pr.slot)}"></i>${pr.id} <b>${f4(v.pnl)}</b> <span style="color:var(--ink2)">funding ${f4(v.fund)} · fees ${v.fees.toFixed(4)}</span></div>`:''}).join('')));
  hit.addEventListener('mouseleave',hideTT)})}
document.getElementById('tbl').innerHTML='<table><tr><th>Día (UTC)</th><th>Par</th><th>PnL</th><th>Funding</th><th>Comisiones</th></tr>'+
 D.days.slice().reverse().flatMap(d=>D.pairs.filter(p=>d.v[p.id]).map(p=>{const v=d.v[p.id];return`<tr><td>${d.date}${d.date===D.today?' *':''}</td><td>${p.id}</td><td>${f4(v.pnl)}</td><td>${f4(v.fund)}</td><td>${v.fees.toFixed(4)}</td></tr>`})).join('')+'</table>';
const draw=()=>{drawLine();drawBars()};draw();let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(draw,120)});
</script></body></html>
"""

if __name__ == "__main__":
    print(build(*(sys.argv[1:3] or [LOG, OUT])))
