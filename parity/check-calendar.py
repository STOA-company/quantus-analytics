"""Replay frozen upstream calendar & day-count vectors without importing Python upstream or accessing the network."""
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

raw = (ROOT / "calendar-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-calendar-parity-v1"
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

max_abs = 0.0
max_rel = 0.0
value_cases = 0
error_cases = 0
doc_bound_cases = 0

for case in data["cases"]:
    header = case["operation"] + "".join(" " + str(v) for v in case["args"])
    rows = [header] + list(case.get("holidays", []))
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

    if case.get("documented_diff") == "gregorian_date_range_bounds":
        doc_bound_cases += 1
        upstream_date = case["output"]["value"]
        assert upstream_date.startswith("0000-") or upstream_date.startswith("10000-"), (
            case["id"],
            upstream_date,
        )
        expected_line = f"ERROR {case['expected_rust_error']}"
        assert output == [expected_line], (case["id"], output, [expected_line])
        continue

    value_cases += 1
    assert len(output) == 1, (case["id"], output)
    actual_str = output[0]
    expected = case["output"]
    kind = expected["kind"]

    if kind == "bool":
        expected_str = "true" if expected["value"] else "false"
        assert actual_str == expected_str, (case["id"], actual_str, expected_str)
    elif kind == "date":
        assert actual_str == expected["value"], (case["id"], actual_str, expected["value"])
    elif kind == "int":
        assert int(actual_str) == expected["value"], (case["id"], actual_str, expected["value"])
    elif kind == "float":
        actual = float(actual_str)
        target = float(expected["value"])
        delta = abs(actual - target)
        assert math.isfinite(actual) and delta <= 1e-12 + 1e-10 * abs(target), (
            case["id"],
            actual,
            target,
        )
        max_abs = max(max_abs, delta)
        if target != 0.0:
            max_rel = max(max_rel, delta / abs(target))
    else:
        raise AssertionError(f"unknown output kind: {kind}")

print(
    f"{len(data['cases'])} calendar cases passed ({value_cases} value, {error_cases} error, "
    f"{doc_bound_cases} documented range-bound diffs); "
    f"max dcf absolute error {max_abs:.3g}; max dcf relative error {max_rel:.3g}"
)
