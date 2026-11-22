"""Replay frozen upstream cov vectors without importing Python upstream or accessing the network."""
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

raw = (ROOT / "cov-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-cov-parity-v1"
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


def exact_two_pass_cov_obs(xs, ys, win_size, ramp):
    out = []
    n = len(xs)
    for idx in range(ramp, n):
        start = max(0, idx + 1 - win_size)
        pairs = [
            (vx, vy)
            for vx, vy in zip(xs[start : idx + 1], ys[start : idx + 1])
            if math.isfinite(vx) and math.isfinite(vy)
        ]
        if len(pairs) < 2:
            out.append(float("nan"))
        else:
            mx = sum(a for a, _ in pairs) / len(pairs)
            my = sum(b for _, b in pairs) / len(pairs)
            sxy = sum((a - mx) * (b - my) for a, b in pairs)
            out.append(sxy / (len(pairs) - 1))
    return out


max_abs = 0.0
max_rel = 0.0
value_cases = 0
error_cases = 0
documented_days_size0_cases = 0
documented_empty_days_ramp_cases = 0
documented_const_residue_cases = 0
documented_large_offset_cases = 0
signed_zero_checks = 0

for case in data["cases"]:
    len_x = len(case["input_x"]["dates"])
    rows = [case["operation"] + "".join(" " + str(v) for v in case["args"]) + f" {len_x}"]
    rows += [
        f"{d} {unpack(v)}"
        for d, v in zip(case["input_x"]["dates"], case["input_x"]["values"])
    ]
    rows += [
        f"{d} {unpack(v)}"
        for d, v in zip(case["input_y"]["dates"], case["input_y"]["values"])
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

    note = case.get("upstream_note")
    if note == "upstream_empty_days_ramp_index_error":
        # Upstream apply_ramp raises IndexError indexing an empty date index with Days ramp > 0;
        # Rust returns an empty Series.
        documented_empty_days_ramp_cases += 1
        assert case["upstream_error"]["upstream_type"] == "IndexError", case["id"]
        assert output == [], (case["id"], output)
        continue

    expected = case["output"]
    valid_index_kinds = (
        ("Index[date]",)
        if expected["dates"]
        else ("RangeIndex", "Index", "Index[date]")
    )
    assert (
        expected["index_kind"] in valid_index_kinds
        and expected["timezone"] is None
    ), case["id"]
    assert expected["dtype"] == "float64", (case["id"], expected["dtype"])
    assert expected["unit"] == "input_product", case["id"]
    assert len(expected["dates"]) == len(expected["values"]), case["id"]

    if note == "upstream_zero_day_offset_all_nan":
        # Upstream DateOffset(days=0) yields all NaN; Rust consistently rejects
        # Window::Days { size: 0, .. } on non-empty x with Error::Window.
        documented_days_size0_cases += 1
        assert all(
            isinstance(v, dict) and v["special"] == "NaN" for v in expected["values"]
        ), case["id"]
        expected_line = f"ERROR {case['rust_error']}"
        assert output == [expected_line], (case["id"], output, [expected_line])
        continue

    if note == "upstream_constant_window_rounding_residue":
        documented_const_residue_cases += 1
        assert len(output) == len(expected["dates"]), (
            case["id"],
            len(output),
            len(expected["dates"]),
        )
        seen_residue = False
        for i, (row, date, raw_value) in enumerate(
            zip(output, expected["dates"], expected["values"])
        ):
            actual_date, value = row.split()
            assert actual_date == date, (case["id"], i, actual_date, date)
            actual = float(value)
            target = float(unpack(raw_value))
            if math.isnan(target):
                assert math.isnan(actual), (case["id"], i, actual)
            else:
                if target != 0.0:
                    assert abs(target) < 1e-12, (case["id"], i, target)
                    seen_residue = True
                assert actual == 0.0 and math.copysign(1.0, actual) == 1.0, (
                    case["id"],
                    i,
                    actual,
                    target,
                )
        assert seen_residue, (case["id"], "expected non-zero upstream rounding residue")
        continue

    if note == "pandas_rolling_cov_large_offset_cancellation":
        documented_large_offset_cases += 1
        assert len(output) == len(expected["dates"]), (
            case["id"],
            len(output),
            len(expected["dates"]),
        )
        assert case["input_x"]["dates"] == case["input_y"]["dates"], case["id"]
        xs = [float(unpack(v)) for v in case["input_x"]["values"]]
        ys = [float(unpack(v)) for v in case["input_y"]["values"]]
        kind, size, ramp = case["args"]
        win_size = len(xs) if kind == "expanding" else int(size)
        exact_vals = exact_two_pass_cov_obs(xs, ys, win_size, int(ramp))
        seen_cancellation = False
        for i, (row, date, raw_target, exact) in enumerate(
            zip(output, expected["dates"], expected["values"], exact_vals)
        ):
            actual_date, value = row.split()
            assert actual_date == date, (case["id"], i, actual_date, date)
            actual = float(value)
            target = float(unpack(raw_target))
            if math.isnan(exact):
                assert math.isnan(actual) and math.isnan(target), (case["id"], i, actual, target)
            else:
                if abs(target - exact) > 1e-6:
                    seen_cancellation = True
                assert math.isfinite(actual) and abs(actual - exact) <= 1e-12 + 1e-10 * abs(exact), (
                    case["id"],
                    i,
                    actual,
                    exact,
                    target,
                )
        assert seen_cancellation, (case["id"], "expected upstream large-offset cancellation")
        continue

    value_cases += 1
    assert len(output) == len(expected["dates"]), (
        case["id"],
        len(output),
        len(expected["dates"]),
    )

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
            if target == 0.0:
                signed_zero_checks += 1
                assert actual == 0.0 and math.copysign(1.0, actual) == math.copysign(
                    1.0, target
                ), (case["id"], i, actual, target)
            max_abs = max(max_abs, delta)
            if target:
                max_rel = max(max_rel, delta / abs(target))

documented_total = (
    documented_days_size0_cases
    + documented_empty_days_ramp_cases
    + documented_const_residue_cases
    + documented_large_offset_cases
)
print(
    f"{len(data['cases'])} cov cases passed ({value_cases} value, {error_cases} error, "
    f"{documented_total} documented diffs: "
    f"{documented_days_size0_cases} days-size0 Error::Window, "
    f"{documented_empty_days_ramp_cases} empty-days-ramp IndexError, "
    f"{documented_const_residue_cases} const rounding residue, "
    f"{documented_large_offset_cases} large-offset cancellation; "
    f"{signed_zero_checks} strict signed-zero checks); "
    f"max absolute error {max_abs:.3g}; max relative error {max_rel:.3g}"
)
