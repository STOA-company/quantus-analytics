"""Replay frozen upstream volatility vectors without importing Python upstream or accessing the network."""
import argparse
import hashlib
import json
import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--replay", action="store_true", required=True)
parser.parse_args()

raw = (ROOT / "volatility-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-volatility-parity-v1"
assert data["upstream_sha"] == "70220e300524d54bd273be0f4c485d53b3780482"
assert (
    hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest()
    == data["python_lock_sha256"]
)

subprocess.run(
    ["cargo", "build", "--offline", "--locked", "--example", "replay"],
    cwd=ROOT.parent,
    check=True,
)
binary = str(ROOT.parent / "target/debug/examples/replay")


def unpack(v):
    if isinstance(v, dict):
        return {"NaN": "NaN", "+Inf": "inf", "-Inf": "-inf"}[v["special"]]
    return str(v)


max_abs = 0.0
max_rel = 0.0
value_cases = 0
error_cases = 0

for case in data["cases"]:
    rows = [case["operation"] + "".join(" " + str(v) for v in case["args"])]
    rows += [
        f"{d} {unpack(v)}"
        for d, v in zip(case["input"]["dates"], case["input"]["values"])
    ]
    output = subprocess.run(
        [binary],
        input="\n".join(rows) + "\n",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    if "error" in case:
        error_cases += 1
        expected_line = f"ERROR {case['error']['rust']}"
        assert output == [expected_line], (case["id"], output, [expected_line])
        continue

    value_cases += 1
    expected = case["output"]
    valid_index_kinds = ("Index[date]",) if expected["dates"] else ("RangeIndex", "Index")
    assert (
        expected["index_kind"] in valid_index_kinds
        and expected["timezone"] is None
    ), case["id"]
    assert expected["dtype"] == "float64", case["id"]
    assert expected["unit"] == "percent", case["id"]
    assert len(output) == len(expected["dates"]), (case["id"], len(output), len(expected["dates"]))

    for i, (row, date, raw_value) in enumerate(
        zip(output, expected["dates"], expected["values"])
    ):
        actual_date, value = row.split()
        assert actual_date == date, (case["id"], i, actual_date, date)
        actual = float(value)
        target = float(unpack(raw_value))
        if math.isnan(target):
            assert math.isnan(actual), (case["id"], i, actual)
        elif math.isinf(target):
            assert actual == target, (case["id"], i, actual, target)
        else:
            delta = abs(actual - target)
            assert math.isfinite(actual) and delta <= 1e-12 + 1e-10 * abs(target), (
                case["id"],
                i,
                actual,
                target,
            )
            max_abs = max(max_abs, delta)
            if target:
                max_rel = max(max_rel, delta / abs(target))

print(
    f"{len(data['cases'])} volatility cases passed ({value_cases} value, {error_cases} error); "
    f"max absolute error {max_abs:.3g}; max relative error {max_rel:.3g}"
)
