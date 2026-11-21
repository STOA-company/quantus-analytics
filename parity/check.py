"""Replay frozen upstream vectors without importing Python upstream or accessing the network."""
import argparse
import hashlib
import json
import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--replay', action='store_true', required=True)
parser.parse_args()
raw = (ROOT/'vectors.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == (ROOT/'vectors.sha256').read_text().split()[0]
data = json.loads(raw)
assert data['schema'] == 'gsquant-parity-v1'
assert hashlib.sha256((ROOT/'python-lock.txt').read_bytes()).hexdigest() == data['python_lock_sha256']
subprocess.run(['cargo','build','--offline','--locked','--example','replay'], cwd=ROOT.parent, check=True)
max_abs = max_rel = 0.0
for case in data['cases']:
    def unpack(v):
        return {'NaN': 'NaN', '+Inf': 'inf', '-Inf': '-inf'}[v['special']] if isinstance(v,dict) else str(v)
    rows = [case['operation'] + ''.join(' '+str(v) for v in case['args'])]
    rows += [d+' '+unpack(v) for d,v in zip(case['input']['dates'],case['input']['values'])]
    output = subprocess.run([str(ROOT.parent/'target/debug/examples/replay')],input='\n'.join(rows)+'\n',text=True,capture_output=True,check=True).stdout.splitlines()
    expected = case['output']
    assert expected['index_kind'] == ('Index[date]' if expected['dates'] else 'RangeIndex') and expected['timezone'] is None
    assert expected['dtype'] == 'float64'
    assert len(output) == len(expected['dates']), case['id']
    for i, (row, date, raw_value) in enumerate(zip(output, expected['dates'], expected['values'])):
        actual_date, value = row.split()
        assert actual_date == date, (case['id'],i,actual_date,date)
        actual = float(value); target = float(unpack(raw_value))
        if math.isnan(target): assert math.isnan(actual), (case['id'],i)
        elif math.isinf(target): assert actual == target, (case['id'],i)
        else:
            delta = abs(actual-target)
            assert math.isfinite(actual) and delta <= 1e-12 + 1e-10*abs(target), (case['id'],i,actual,target)
            max_abs=max(max_abs,delta)
            if target: max_rel=max(max_rel,delta/abs(target))
print(f"{len(data['cases'])} cases passed; max absolute error {max_abs:.3g}; max relative error {max_rel:.3g}")
