"""Generate upstream vectors only. Run with the locked upstream Python environment."""
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys

# The caller supplies the fixed checkout; no network or Rust execution here.
sys.path.insert(0, sys.argv[1])
import pandas as pd
from gs_quant.timeseries import moving_average, returns, Window, Returns

ROOT = pathlib.Path(__file__).resolve().parent
D = [dt.date(2019, 1, i) for i in range(1, 7)]
prices = [3., 2., 3., 1., 3., 6.]
levels = [100., 101., 103.02, 100.9596, 100.9596, 102.978792]
cases = []
def add(id, op, args, dates, values, fn):
    series = pd.Series(values, index=dates, dtype=float) if dates else pd.Series(dtype=float)
    out = fn(series)
    def val(v):
        if math.isnan(v): return {"special": "NaN"}
        if math.isinf(v): return {"special": "+Inf" if v > 0 else "-Inf"}
        return float(v)
    cases.append(dict(id=id, node_id=("gs_quant/test/timeseries/test_econometrics.py::test_returns" if op == "returns" else "gs_quant/test/timeseries/test_technicals.py::test_moving_average"), operation=op, args=args, input={"index_kind": "Index[date]" if dates else type(series.index).__name__, "timezone": None, "dates": [str(d) for d in dates], "values": [val(v) for v in values], "unit": "price", "dtype": "float64"}, output={"index_kind": "Index[date]" if len(out) else type(out.index).__name__, "timezone": None, "dates": [str(d) for d in out.index], "values": [val(v) for v in out], "unit": "price" if op != "returns" or args[1] == "absolute" else "ratio", "dtype": "float64"}))
add('ma_default','expanding',[],D,prices,lambda x:moving_average(x))
for n in (1,2): add(f'ma_{n}','observations',[n,0],D,prices,lambda x,n=n:moving_average(x,Window(n,0)))
add('ma_2d','days',[2,2],D,prices,lambda x:moving_average(x,'2d'))
add('ma_empty','days',[2,2],[],[],lambda x:moving_average(x,'2d'))
add('returns_empty','returns',[1,'simple'],[],[],lambda x:returns(x))
add('returns_default','returns',[1,'simple'],D,levels,lambda x:returns(x))
for kind in ('simple','logarithmic','absolute'):
    for lag in (1,2): add(f'returns_{kind}_{lag}','returns',[lag,kind],D,levels,lambda x,lag=lag,kind=kind:returns(x,lag,getattr(Returns,kind.upper())))
add('ma_nan','observations',[2,0],D,[3.,float('nan'),float('nan'),1.,3.,6.],lambda x:moving_average(x,Window(2,0)))
add('returns_nan','returns',[1,'simple'],D,[100.,float('nan'),103.02,100.9596,100.9596,102.978792],lambda x:returns(x))
payload = {"schema": "gsquant-parity-v1", "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482", "python": sys.version.split()[0], "pandas": pd.__version__, "python_lock_sha256": hashlib.sha256((ROOT/"python-lock.txt").read_bytes()).hexdigest(), "source": ["gs_quant/timeseries/technicals.py:moving_average", "gs_quant/timeseries/econometrics.py:returns"], "cases": cases}
raw=(json.dumps(payload,indent=2,allow_nan=False)+'\n').encode()
(ROOT/'vectors.json').write_bytes(raw)
(ROOT/'vectors.sha256').write_text(hashlib.sha256(raw).hexdigest()+'  vectors.json\n')
print(f'generated {len(cases)} upstream cases')
