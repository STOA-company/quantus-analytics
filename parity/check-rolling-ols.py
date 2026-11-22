"""Replay frozen upstream RollingLinearRegression vectors without importing Python upstream or accessing the network."""
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

raw = (ROOT / "rolling-ols-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-rolling-ols-parity-v1"
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

ATOL = 1e-10
RTOL = 1e-8


def unpack(v):
    if isinstance(v, dict):
        return {"NaN": "NaN", "+Inf": "inf", "-Inf": "-inf"}[v["special"]]
    return str(v)


def parse_replay_output(lines):
    assert lines, "empty replay output"
    if len(lines) == 1 and lines[0].startswith("ERROR "):
        return {"error": lines[0].split(" ", 1)[1]}
    assert lines[0].startswith("COLS "), lines[0]
    cols = [int(tok) for tok in lines[0].split()[1:]]
    idx = 1
    parsed = {"column_indices": cols, "coefficients": {}}
    section = None
    cur_dates = None
    cur_vals = None

    def flush():
        nonlocal section, cur_dates, cur_vals
        if section is None:
            return
        if isinstance(section, tuple) and section[0] == "COEF":
            parsed["coefficients"][section[1]] = (cur_dates, cur_vals)
        else:
            parsed[section] = (cur_dates, cur_vals)

    while idx < len(lines):
        line = lines[idx]
        if line.startswith("COEF "):
            flush()
            section = ("COEF", int(line.split()[1]))
            cur_dates, cur_vals = [], []
        elif line in ("R_SQUARED", "FITTED", "STD_ERR"):
            flush()
            section = line
            cur_dates, cur_vals = [], []
        else:
            d_str, v_str = line.split()
            cur_dates.append(d_str)
            cur_vals.append(float(v_str))
        idx += 1
    flush()
    return parsed


def compare_series(case_id, label, actual_dates, actual_vals, expected_dates, expected_raw, stats):
    assert actual_dates == expected_dates, (case_id, label, actual_dates, expected_dates)
    assert len(actual_vals) == len(expected_raw), (
        case_id,
        label,
        len(actual_vals),
        len(expected_raw),
    )
    for i, (actual, raw_val) in enumerate(zip(actual_vals, expected_raw)):
        target = float(unpack(raw_val))
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
doc_const_y_cases = 0
doc_ill_cond_cases = 0
doc_singular_leak_cases = 0
doc_short_aligned_cases = 0
doc_empty_pred_cases = 0

for case in data["cases"]:
    window, fit_intercept = case["args"]
    inputs_x = case["inputs_x"]
    input_y = case["input_y"]
    header = (
        f"rolling_ols {window} {fit_intercept} {len(inputs_x)}"
        + "".join(f" {len(sx['dates'])}" for sx in inputs_x)
    )
    rows = [header]
    for sx in inputs_x:
        rows += [f"{d} {unpack(v)}" for d, v in zip(sx["dates"], sx["values"])]
    rows += [f"{d} {unpack(v)}" for d, v in zip(input_y["dates"], input_y["values"])]

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

    note = case.get("upstream_note")
    if note == "upstream_short_aligned_error":
        doc_short_aligned_cases += 1
        assert case["upstream_error"]["upstream_type"] in ("ValueError", "IndexError"), case["id"]
        expected_line = f"ERROR {case['rust_error']}"
        assert stdout_lines == [expected_line], (case["id"], stdout_lines, [expected_line])
        continue

    if note == "upstream_empty_predictors_value_error":
        doc_empty_pred_cases += 1
        assert case["upstream_error"]["upstream_type"] == "ValueError", case["id"]
        expected_line = f"ERROR {case['rust_error']}"
        assert stdout_lines == [expected_line], (case["id"], stdout_lines, [expected_line])
        continue

    expected = case["output"]
    assert expected["index_kind"] == "Index[date]" and expected["timezone"] is None, case["id"]
    assert expected["dtype"] == "float64", (case["id"], expected["dtype"])

    parsed = parse_replay_output(stdout_lines)
    assert "error" not in parsed, (case["id"], parsed)
    assert parsed["column_indices"] == expected["column_indices"], (
        case["id"],
        parsed["column_indices"],
        expected["column_indices"],
    )

    exp_dates = expected["dates"]

    if note == "statsmodels_const_y_residual_noise":
        doc_const_y_cases += 1
        y0 = float(unpack(input_y["values"][0]))
        w = int(window)
        # Verify frozen upstream vector records the constant-y residual noise signature
        assert any(
            (isinstance(v, dict) and v["special"] == "-Inf")
            for v in expected["r_squared"][w - 1 :]
        ), (case["id"], "expected -Inf in frozen upstream r_squared")
        # Verify Rust emits exact beta=[y0, 0.0], fitted=y0, r_squared=NaN, std_err=+0.0
        for c in parsed["column_indices"]:
            d_c, v_c = parsed["coefficients"][c]
            assert d_c == exp_dates, (case["id"], c)
            target_b = y0 if c == 0 else 0.0
            for i in range(w - 1):
                assert math.isnan(v_c[i]), (case["id"], c, i, v_c[i])
            for i in range(w - 1, len(exp_dates)):
                assert v_c[i] == target_b, (case["id"], c, i, v_c[i], target_b)
        d_r2, v_r2 = parsed["R_SQUARED"]
        assert d_r2 == exp_dates and all(math.isnan(v) for v in v_r2), (case["id"], v_r2)
        d_fit, v_fit = parsed["FITTED"]
        assert d_fit == exp_dates
        for i in range(w - 1):
            assert math.isnan(v_fit[i])
        for i in range(w - 1, len(exp_dates)):
            assert v_fit[i] == y0, (case["id"], i, v_fit[i], y0)
        d_se, v_se = parsed["STD_ERR"]
        assert d_se == exp_dates
        for i in range(w - 1):
            assert math.isnan(v_se[i])
        for i in range(w - 1, len(exp_dates)):
            assert v_se[i] == 0.0 and math.copysign(1.0, v_se[i]) == 1.0, (case["id"], i, v_se[i])
        continue

    if note == "statsmodels_normal_eq_ill_conditioned":
        doc_ill_cond_cases += 1
        w = int(window)
        exact_intercept = -19_999_999.0
        exact_slope = 2.0
        y_vals = [float(unpack(v)) for v in input_y["values"]]
        # Verify frozen upstream normal-equation vector loses digits on intercept
        up_b0 = [float(unpack(v)) for v in expected["coefficients"]["0"][w - 1 :]]
        assert any(abs(v - exact_intercept) > 1e2 for v in up_b0), (
            case["id"],
            "expected normal-equation intercept loss in upstream vector",
            up_b0,
        )
        # Verify Rust Householder QR recovers accurate coefficients, R^2=1.0, fitted=y, std_err~0
        d_b0, v_b0 = parsed["coefficients"][0]
        d_b1, v_b1 = parsed["coefficients"][1]
        d_r2, v_r2 = parsed["R_SQUARED"]
        d_fit, v_fit = parsed["FITTED"]
        d_se, v_se = parsed["STD_ERR"]
        assert d_b0 == d_b1 == d_r2 == d_fit == d_se == exp_dates, case["id"]
        for i in range(w - 1):
            assert (
                math.isnan(v_b0[i])
                and math.isnan(v_b1[i])
                and math.isnan(v_r2[i])
                and math.isnan(v_fit[i])
                and math.isnan(v_se[i])
            ), (case["id"], i)
        for i in range(w - 1, len(exp_dates)):
            assert abs(v_b0[i] - exact_intercept) <= ATOL + RTOL * abs(exact_intercept), (
                case["id"],
                i,
                v_b0[i],
                exact_intercept,
            )
            assert abs(v_b1[i] - exact_slope) <= ATOL + RTOL * abs(exact_slope), (
                case["id"],
                i,
                v_b1[i],
                exact_slope,
            )
            assert abs(v_r2[i] - 1.0) <= ATOL, (case["id"], i, v_r2[i])
            assert abs(v_fit[i] - y_vals[i]) <= 1e-7, (
                case["id"],
                i,
                v_fit[i],
                y_vals[i],
            )
            assert v_se[i] <= 1e-7, (case["id"], i, v_se[i])
        continue

    if note == "statsmodels_singular_normal_eq_leak":
        doc_singular_leak_cases += 1
        w = int(window)
        # Verify frozen upstream vector records at least one non-NaN leaked value post-warmup
        up_post_warmup = [
            float(unpack(v))
            for c in expected["column_indices"]
            for v in expected["coefficients"][str(c)][w - 1 :]
        ] + [
            float(unpack(v))
            for key in ("r_squared", "fitted_values", "standard_deviation_of_errors")
            for v in expected[key][w - 1 :]
        ]
        assert any(not math.isnan(v) for v in up_post_warmup), (
            case["id"],
            "expected at least one non-NaN leaked value in frozen upstream vector",
        )
        # Verify Rust emits NaN for every date across all outputs with exact matching dates
        for c in parsed["column_indices"]:
            d_c, v_c = parsed["coefficients"][c]
            assert d_c == exp_dates and all(math.isnan(v) for v in v_c), (
                case["id"],
                f"coef[{c}]",
                d_c,
                v_c,
            )
        for key in ("R_SQUARED", "FITTED", "STD_ERR"):
            d_s, v_s = parsed[key]
            assert d_s == exp_dates and all(math.isnan(v) for v in v_s), (
                case["id"],
                key,
                d_s,
                v_s,
            )
        continue

    value_cases += 1
    for c in expected["column_indices"]:
        d_c, v_c = parsed["coefficients"][c]
        compare_series(
            case["id"], f"coef[{c}]", d_c, v_c, exp_dates, expected["coefficients"][str(c)], stats
        )
    compare_series(
        case["id"], "r_squared", *parsed["R_SQUARED"], exp_dates, expected["r_squared"], stats
    )
    compare_series(
        case["id"], "fitted_values", *parsed["FITTED"], exp_dates, expected["fitted_values"], stats
    )
    compare_series(
        case["id"],
        "standard_deviation_of_errors",
        *parsed["STD_ERR"],
        exp_dates,
        expected["standard_deviation_of_errors"],
        stats,
    )

doc_total = (
    doc_const_y_cases
    + doc_ill_cond_cases
    + doc_singular_leak_cases
    + doc_short_aligned_cases
    + doc_empty_pred_cases
)
print(
    f"{len(data['cases'])} rolling_ols cases passed ({value_cases} value, {error_cases} error, "
    f"{doc_total} documented diffs: "
    f"{doc_const_y_cases} const-y residual noise, "
    f"{doc_ill_cond_cases} ill-conditioned 1e7 QR vs normal eq, "
    f"{doc_singular_leak_cases} singular normal-eq leak, "
    f"{doc_short_aligned_cases} short/empty aligned Error::Window, "
    f"{doc_empty_pred_cases} empty predictors Error::Length); "
    f"max absolute error {stats['max_abs']:.3g}; max relative error {stats['max_rel']:.3g}"
)
