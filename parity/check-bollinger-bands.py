"""Replay frozen upstream bollinger_bands vectors without importing Python upstream or accessing the network."""
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

raw = (ROOT / "bollinger-bands-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-bollinger-bands-parity-v1"
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
documented_days_size0_cases = 0
documented_mixed_zero_cases = 0
documented_const_days_inf_cases = 0
signed_zero_checks = 0

for case in data["cases"]:
    header_args = [
        v if isinstance(v, str) else unpack(v) for v in case["args"]
    ]
    rows = [case["operation"] + "".join(" " + str(v) for v in header_args)]
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
    assert expected["dtypes"] == ["float64", "float64"], (case["id"], expected["dtypes"])
    assert expected["unit"] == "input", case["id"]
    assert len(expected["dates"]) == len(expected["lower"]) == len(expected["upper"]), case["id"]

    if case.get("upstream_note") == "upstream_zero_day_offset_fallthrough":
        # Upstream DateOffset(days=0) falls through rolling_std as an expanding window;
        # Rust consistently rejects Window::Days { size: 0, .. } with Error::Window.
        documented_days_size0_cases += 1
        expected_line = f"ERROR {case['rust_error']}"
        assert output == [expected_line], (case["id"], output, [expected_line])
        continue

    mixed_zero_note = (
        case.get("upstream_note") == "pandas_rolling_mean_mixed_signed_zero_gh42064"
    )
    const_days_inf_note = (
        case.get("upstream_note") == "numpy_calendar_const_std_rounding_residue_k_inf"
    )
    if mixed_zero_note:
        documented_mixed_zero_cases += 1
    elif const_days_inf_note:
        documented_const_days_inf_cases += 1
    else:
        value_cases += 1

    assert len(output) == len(expected["dates"]), (
        case["id"],
        len(output),
        len(expected["dates"]),
    )

    seen_mixed_zero_diff = False
    seen_const_days_inf_diff = False
    for i, (row, date, raw_lower, raw_upper) in enumerate(
        zip(output, expected["dates"], expected["lower"], expected["upper"])
    ):
        actual_date, lower_str, upper_str = row.split()
        assert actual_date == date, (case["id"], i, actual_date, date)
        for band_name, raw_actual, raw_target in (
            ("lower", lower_str, raw_lower),
            ("upper", upper_str, raw_upper),
        ):
            actual = float(raw_actual)
            target = float(unpack(raw_target))
            if math.isnan(target):
                assert math.isnan(actual), (case["id"], i, band_name, actual)
            elif math.isinf(target):
                if const_days_inf_note:
                    # Documented NumPy 2.2.6 rolling_std mean-rounding residue (~9.36e-16) on
                    # constant 4.2 x 10 days amplified by k = ±Inf; Rust reuses std's exact +0.0
                    # contract so 0.0 * ±Inf = NaN.
                    assert math.isnan(actual), (case["id"], i, band_name, actual, target)
                    seen_const_days_inf_diff = True
                else:
                    assert actual == target, (case["id"], i, band_name, actual, target)
            else:
                delta = abs(actual - target)
                assert math.isfinite(actual) and delta <= 1e-12 + 1e-10 * abs(target), (
                    case["id"],
                    i,
                    band_name,
                    actual,
                    target,
                )
                if target == 0.0:
                    if mixed_zero_note and band_name == "lower":
                        # Documented pandas 2.3.3 rolling().mean() GH#42064 equal-value fast path
                        # adopts -0.0 on [+0.0, -0.0]; Rust IEEE-754 +0.0 + (-0.0) = +0.0 emits +0.0
                        # (numeric diff 0).
                        assert math.copysign(1.0, target) == -1.0, (case["id"], i, target)
                        assert actual == 0.0 and math.copysign(1.0, actual) == 1.0, (
                            case["id"],
                            i,
                            band_name,
                            actual,
                            target,
                        )
                        seen_mixed_zero_diff = True
                    else:
                        if not mixed_zero_note:
                            signed_zero_checks += 1
                        assert actual == 0.0 and math.copysign(1.0, actual) == math.copysign(
                            1.0, target
                        ), (case["id"], i, band_name, actual, target)
                max_abs = max(max_abs, delta)
                if target:
                    max_rel = max(max_rel, delta / abs(target))

    if mixed_zero_note:
        assert seen_mixed_zero_diff, (case["id"], "expected documented mixed-signed-zero diff")
    if const_days_inf_note:
        assert seen_const_days_inf_diff, (case["id"], "expected documented const-days-inf diff")

documented_total = (
    documented_days_size0_cases
    + documented_mixed_zero_cases
    + documented_const_days_inf_cases
)
print(
    f"{len(data['cases'])} bollinger_bands cases passed ({value_cases} value, {error_cases} error, "
    f"{documented_total} documented diffs: "
    f"{documented_days_size0_cases} days-size0 Error::Window, "
    f"{documented_mixed_zero_cases} mixed signed-zero GH#42064, "
    f"{documented_const_days_inf_cases} const calendar k=±Inf NumPy residue; "
    f"{signed_zero_checks} strict signed-zero checks); "
    f"max absolute error {max_abs:.3g}; max relative error {max_rel:.3g}"
)
