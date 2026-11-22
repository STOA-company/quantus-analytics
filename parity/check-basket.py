"""Replay frozen upstream backtest_basket / basket_series vectors without importing Python upstream or accessing the network."""
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

raw = (ROOT / "basket-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-basket-parity-v1"
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

ATOL = 1e-12
RTOL = 1e-10


def unpack_str(v) -> str:
    if v == "NaN":
        return "NaN"
    if v == "Infinity":
        return "inf"
    if v == "-Infinity":
        return "-inf"
    return repr(float(v))


def unpack_float(v) -> float:
    if v == "NaN":
        return float("nan")
    if v == "Infinity":
        return float("inf")
    if v == "-Infinity":
        return float("-inf")
    return float(v)


def encode_series_lines(spec) -> list:
    dates = spec["dates"]
    values = spec["values"]
    out = [f"SERIES {len(dates)}"]
    out.extend(f"{d} {unpack_str(v)}" for d, v in zip(dates, values))
    return out


def encode_param_lines(param) -> list:
    if param["kind"] == "fixed":
        return [f"FIXED {unpack_str(param['value'])}"]
    if param["kind"] == "series":
        return encode_series_lines(param)
    raise AssertionError(f"unknown param kind: {param['kind']}")


def parse_backtest_output(lines):
    assert lines, "empty replay output"
    if len(lines) == 1 and lines[0].startswith("ERROR "):
        return {"error": lines[0].split(" ", 1)[1]}
    assert lines[0] == "NAV", lines[0]
    section = "NAV"
    nav_dates, nav_vals = [], []
    weights = {}
    cur_dates, cur_vals = nav_dates, nav_vals
    for line in lines[1:]:
        if line.startswith("WEIGHT "):
            idx = int(line.split()[1])
            cur_dates, cur_vals = [], []
            weights[idx] = (cur_dates, cur_vals)
            section = f"WEIGHT {idx}"
        else:
            d_str, v_str = line.split()
            cur_dates.append(d_str)
            cur_vals.append(float(v_str))
    assert section is not None
    return {"nav": (nav_dates, nav_vals), "actual_weights": weights}


def compare_series(case_id, label, actual_dates, actual_vals, expected_dates, expected_raw, stats):
    assert actual_dates == expected_dates, (case_id, label, actual_dates, expected_dates)
    assert len(actual_vals) == len(expected_raw), (
        case_id,
        label,
        len(actual_vals),
        len(expected_raw),
    )
    for i, (actual, raw_val) in enumerate(zip(actual_vals, expected_raw)):
        target = unpack_float(raw_val)
        if math.isnan(target):
            assert math.isnan(actual), (case_id, label, i, actual)
        elif math.isinf(target):
            assert actual == target, (case_id, label, i, actual, target)
        else:
            delta = abs(actual - target)
            assert math.isfinite(actual) and delta <= ATOL + RTOL * abs(target), (
                case_id,
                label,
                i,
                actual,
                target,
            )
            stats["max_abs"] = max(stats["max_abs"], delta)
            if target != 0.0:
                stats["max_rel"] = max(stats["max_rel"], delta / abs(target))


stats = {"max_abs": 0.0, "max_rel": 0.0}
value_cases = 0
error_cases = 0
doc_empty_slice_cases = 0

for case in data["cases"]:
    op = case["operation"]
    rebal = case["rebal_freq"]
    series_list = case["series"]
    weights_list = case["weights"]
    costs_list = case["costs"]

    w_spec = "none" if weights_list is None else str(len(weights_list))
    c_spec = "none" if costs_list is None else str(len(costs_list))
    rows = [f"{op} {rebal} {len(series_list)} {w_spec} {c_spec}"]
    for s in series_list:
        rows.extend(encode_series_lines(s))
    if weights_list is not None:
        for p in weights_list:
            rows.extend(encode_param_lines(p))
    if costs_list is not None:
        for p in costs_list:
            rows.extend(encode_param_lines(p))

    stdout_lines = subprocess.run(
        [binary],
        input="\n".join(rows) + "\n",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    if "error" in case:
        error_cases += 1
        expected_line = f"ERROR {case['error']['rust']}"
        assert stdout_lines == [expected_line], (case["id"], stdout_lines, [expected_line])
        continue

    if case.get("documented_diff") == "empty_slice_vs_python_falsy_default":
        doc_empty_slice_cases += 1
        assert "output" in case and len(case["output"]["dates"]) > 0, case["id"]
        expected_line = f"ERROR {case['expected_rust_error']}"
        assert stdout_lines == [expected_line], (case["id"], stdout_lines, [expected_line])
        continue

    value_cases += 1
    expected = case["output"]
    exp_dates = expected["dates"]

    if op == "backtest_basket":
        parsed = parse_backtest_output(stdout_lines)
        assert "error" not in parsed, (case["id"], parsed)
        nav_dates, nav_vals = parsed["nav"]
        compare_series(case["id"], "nav", nav_dates, nav_vals, exp_dates, expected["nav"], stats)
        exp_aw = expected["actual_weights"]
        assert len(parsed["actual_weights"]) == len(exp_aw), (
            case["id"],
            len(parsed["actual_weights"]),
            len(exp_aw),
        )
        for a, exp_col in enumerate(exp_aw):
            aw_dates, aw_vals = parsed["actual_weights"][a]
            compare_series(case["id"], f"actual_weights[{a}]", aw_dates, aw_vals, exp_dates, exp_col, stats)
    elif op == "basket_series":
        assert not (len(stdout_lines) == 1 and stdout_lines[0].startswith("ERROR ")), (
            case["id"],
            stdout_lines,
        )
        act_dates, act_vals = [], []
        for line in stdout_lines:
            d_str, v_str = line.split()
            act_dates.append(d_str)
            act_vals.append(float(v_str))
        compare_series(case["id"], "basket_series", act_dates, act_vals, exp_dates, expected["values"], stats)
    else:
        raise AssertionError(f"unknown operation: {op}")

print(
    f"{len(data['cases'])} basket cases passed ({value_cases} value, {error_cases} error, "
    f"{doc_empty_slice_cases} documented empty-slice diffs); "
    f"max absolute error {stats['max_abs']:.3g}; max relative error {stats['max_rel']:.3g}"
)
