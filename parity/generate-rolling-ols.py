"""Generate upstream RollingLinearRegression parity vectors only. Run with the locked upstream Python environment."""
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import warnings

# The caller supplies the fixed checkout; no network or Rust execution here.
sys.path.insert(0, sys.argv[1])
import numpy as np
import pandas as pd
import statsmodels as sm
from gs_quant.timeseries.statistics import RollingLinearRegression

ROOT = pathlib.Path(__file__).resolve().parent

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")

UPSTREAM_X1_DATES = [dt.date(2019, 1, d) for d in range(1, 8)]
UPSTREAM_X1_VALS = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, NAN]
UPSTREAM_X2_DATES = [dt.date(2019, 1, d) for d in range(1, 9)]
UPSTREAM_X2_VALS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
UPSTREAM_Y_DATES = [dt.date(2019, 1, d) for d in range(1, 8)]
UPSTREAM_Y_VALS = [10.0, 14.0, 20.0, 28.0, 28.0, 40.0, 60.0]

D5 = [dt.date(2020, 1, d) for d in range(1, 6)]
D6 = [dt.date(2020, 1, d) for d in range(1, 7)]
D7 = [dt.date(2020, 1, d) for d in range(1, 8)]
D8 = [dt.date(2020, 5, d) for d in range(1, 9)]

LEAP_DATES = [
    dt.date(2020, 2, 26),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 5),
    dt.date(2020, 3, 9),
]

X6_LIN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
X6_QUAD = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
X6_ALT = [2.0, 1.0, 4.0, 2.0, 5.0, 3.0]
Y6_NOISY = [12.0, 14.0, 19.0, 24.0, 33.0, 42.0]

F7_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
F7_2 = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 1.0]
F7_3 = [2.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0]
Y7_3F = [9.0, 11.0, 18.0, 22.0, 29.0, 27.0, 34.0]


def pack(v: float):
    if math.isnan(v):
        return {"special": "NaN"}
    if math.isinf(v):
        return {"special": "+Inf" if v > 0 else "-Inf"}
    return float(v)


def fmt_date(d) -> str:
    if isinstance(d, pd.Timestamp):
        return d.date().isoformat()
    if isinstance(d, dt.date):
        return d.isoformat()
    return str(d)


def index_kind(idx) -> str:
    if len(idx) > 0 and isinstance(idx[0], dt.date) and not isinstance(idx[0], dt.datetime):
        return "Index[date]"
    return type(idx).__name__


def pack_series(dates, values, series):
    return {
        "index_kind": index_kind(series.index),
        "timezone": None,
        "dates": [d.isoformat() for d in dates],
        "values": [pack(v) for v in values],
        "dtype": "float64",
    }


cases = []


def add_case(
    case_id,
    source_ref,
    window,
    fit_intercept,
    x_specs,
    y_dates,
    y_values,
    *,
    pass_single_series=False,
    rust_error=None,
    upstream_note=None,
):
    sx_list = [
        pd.Series(vals, index=dates, dtype=float) if dates else pd.Series(dtype=float)
        for dates, vals in x_specs
    ]
    sy = pd.Series(y_values, index=y_dates, dtype=float) if y_dates else pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "rolling_ols",
        "args": [window, "true" if fit_intercept else "false"],
        "inputs_x": [
            pack_series(dates, vals, sx) for (dates, vals), sx in zip(x_specs, sx_list)
        ],
        "input_y": pack_series(y_dates, y_values, sy),
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note
        if rust_error is not None:
            entry["rust_error"] = rust_error

    x_arg = sx_list[0] if (pass_single_series and len(sx_list) == 1) else sx_list
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg = RollingLinearRegression(x_arg, sy, window, fit_intercept=fit_intercept)
            col_indices = [int(c) for c in reg._res.params.columns]
            r2 = reg.r_squared()
            fitted = reg.fitted_values()
            stderr = reg.standard_deviation_of_errors()
            coeffs = {str(c): reg.coefficient(c) for c in col_indices}
        except Exception as exc:
            if upstream_note in (
                "upstream_short_aligned_error",
                "upstream_empty_predictors_value_error",
            ):
                entry["upstream_error"] = {
                    "upstream_type": type(exc).__name__,
                    "upstream_message": str(exc),
                }
            elif rust_error is not None and upstream_note is None:
                entry["error"] = {
                    "rust": rust_error,
                    "upstream_type": type(exc).__name__,
                    "upstream_message": str(exc),
                }
            else:
                raise
        else:
            if upstream_note in (
                "upstream_short_aligned_error",
                "upstream_empty_predictors_value_error",
            ):
                raise AssertionError(f"{case_id}: expected upstream error, got output")
            if rust_error is not None and upstream_note is None:
                raise AssertionError(
                    f"{case_id}: expected upstream error for {rust_error}, got output"
                )
            dates = [fmt_date(d) for d in r2.index]
            entry["output"] = {
                "column_indices": col_indices,
                "index_kind": index_kind(r2.index),
                "timezone": None,
                "dates": dates,
                "coefficients": {
                    k: [pack(float(v)) for v in s] for k, s in coeffs.items()
                },
                "r_squared": [pack(float(v)) for v in r2],
                "fitted_values": [pack(float(v)) for v in fitted],
                "standard_deviation_of_errors": [pack(float(v)) for v in stderr],
                "dtype": str(r2.dtype),
            }
    cases.append(entry)


# 1. Upstream test_statistics.py::test_rolling_linear_regression (2 cases)
add_case(
    "upstream-test-rolling-linear-regression",
    "test_statistics.py::test_rolling_linear_regression",
    4,
    True,
    [(UPSTREAM_X1_DATES, UPSTREAM_X1_VALS), (UPSTREAM_X2_DATES, UPSTREAM_X2_VALS)],
    UPSTREAM_Y_DATES,
    UPSTREAM_Y_VALS,
)
add_case(
    "upstream-test-no-intercept",
    "test_statistics.py::test_rolling_linear_regression",
    4,
    False,
    [(UPSTREAM_X1_DATES, UPSTREAM_X1_VALS), (UPSTREAM_X2_DATES, UPSTREAM_X2_VALS)],
    UPSTREAM_Y_DATES,
    UPSTREAM_Y_VALS,
)

# 2. 1 / 2 / 3 predictor models with fit_intercept true & false (11 cases)
add_case(
    "pred1-intercept-w3",
    "synthetic-multifactor",
    3,
    True,
    [(D6, X6_LIN)],
    D6,
    [3.0, 5.0, 7.0, 11.0, 11.0, 13.0],
    pass_single_series=True,
)
add_case(
    "pred1-intercept-w4",
    "synthetic-multifactor",
    4,
    True,
    [(D6, X6_QUAD)],
    D6,
    Y6_NOISY,
)
add_case(
    "pred1-no-intercept-w2",
    "synthetic-multifactor",
    2,
    False,
    [(D6, X6_LIN)],
    D6,
    [3.0, 5.0, 7.0, 11.0, 11.0, 13.0],
    pass_single_series=True,
)
add_case(
    "pred1-no-intercept-w4",
    "synthetic-multifactor",
    4,
    False,
    [(D6, X6_QUAD)],
    D6,
    [-3.0, -5.0, -9.0, -16.0, -24.0, -35.0],
)
add_case(
    "pred2-intercept-w4",
    "synthetic-multifactor",
    4,
    True,
    [(D6, X6_QUAD), (D6, X6_ALT)],
    D6,
    Y6_NOISY,
)
add_case(
    "pred2-intercept-w5",
    "synthetic-multifactor",
    5,
    True,
    [(D6, X6_QUAD), (D6, X6_ALT)],
    D6,
    Y6_NOISY,
)
add_case(
    "pred2-no-intercept-w3",
    "synthetic-multifactor",
    3,
    False,
    [(D6, X6_QUAD), (D6, X6_ALT)],
    D6,
    Y6_NOISY,
)
add_case(
    "pred3-intercept-w5",
    "synthetic-multifactor",
    5,
    True,
    [(D7, F7_1), (D7, F7_2), (D7, F7_3)],
    D7,
    Y7_3F,
)
add_case(
    "pred3-intercept-w6",
    "synthetic-multifactor",
    6,
    True,
    [(D7, F7_1), (D7, F7_2), (D7, F7_3)],
    D7,
    Y7_3F,
)
add_case(
    "pred3-no-intercept-w4",
    "synthetic-multifactor",
    4,
    False,
    [(D7, F7_1), (D7, F7_2), (D7, F7_3)],
    D7,
    Y7_3F,
)
add_case(
    "exact-window-eq-len",
    "synthetic-multifactor",
    7,
    True,
    [(D7, F7_1), (D7, F7_2), (D7, F7_3)],
    D7,
    Y7_3F,
)

# 3. sm.add_constant skip & has_constant centered vs uncentered R^2 (7 cases)
add_case(
    "const-skip-pred1-const5",
    "synthetic-constant-handling",
    3,
    True,
    [(D6, X6_QUAD), (D6, [5.0] * 6)],
    D6,
    Y6_NOISY,
)
add_case(
    "const-skip-pred0-const-neg2",
    "synthetic-constant-handling",
    4,
    True,
    [(D6, [-2.0] * 6), (D6, X6_QUAD)],
    D6,
    Y6_NOISY,
)
add_case(
    "const-no-skip-zero-pred",
    "synthetic-constant-handling",
    4,
    True,
    [(D6, X6_QUAD), (D6, [0.0] * 6)],
    D6,
    Y6_NOISY,
)
add_case(
    "const-no-skip-nan-in-const",
    "synthetic-constant-handling",
    4,
    True,
    [(D6, X6_QUAD), (D6, [5.0, 5.0, NAN, 5.0, 5.0, 5.0])],
    D6,
    Y6_NOISY,
)
add_case(
    "const-no-skip-diff-outside-y",
    "synthetic-constant-handling",
    4,
    True,
    [(D6, X6_QUAD), (D7, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0])],
    D6,
    Y6_NOISY,
    upstream_note="statsmodels_singular_normal_eq_leak",
)
add_case(
    "no-intercept-const-pred-centered-r2",
    "synthetic-constant-handling",
    4,
    False,
    [(D6, X6_QUAD), (D6, [5.0] * 6)],
    D6,
    Y6_NOISY,
)
add_case(
    "no-intercept-const-on-aligned-only",
    "synthetic-constant-handling",
    4,
    False,
    [(D6, X6_QUAD), (D7, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0])],
    D6,
    Y6_NOISY,
)

# 4. Window-local constant & collinear rank deficiency (4 cases)
add_case(
    "rank-def-local-const-mid",
    "synthetic-rank-deficiency",
    4,
    True,
    [(D6, [2.0, 2.0, 2.0, 2.0, 5.0, 9.0])],
    D6,
    Y6_NOISY,
)
add_case(
    "rank-def-collinear-all",
    "synthetic-rank-deficiency",
    4,
    True,
    [(D6, X6_QUAD), (D6, [2.0 * v + 3.0 for v in X6_QUAD])],
    D6,
    Y6_NOISY,
    upstream_note="statsmodels_singular_normal_eq_leak",
)
add_case(
    "rank-def-collinear-partial",
    "synthetic-rank-deficiency",
    4,
    True,
    [(D6, X6_QUAD), (D6, [2.0, 4.0, 8.0, 14.0, 30.0, 50.0])],
    D6,
    Y6_NOISY,
)
add_case(
    "rank-def-zero-pred-no-intercept",
    "synthetic-rank-deficiency",
    3,
    False,
    [(D6, [0.0, 0.0, 0.0, 0.0, 2.0, 4.0])],
    D6,
    [1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
)

# 5. NaN / ±Inf filtering, misaligned dates, leap day 2020-02-29, negative values (6 cases)
add_case(
    "filter-nan-and-inf-x-and-y",
    "synthetic-filtering-alignment",
    4,
    True,
    [
        (D8, [1.0, NAN, 2.0, 3.0, INF, 5.0, 7.0, 9.0]),
        (D8, [2.0, 1.0, 5.0, NINF, 4.0, 3.0, 8.0, 6.0]),
    ],
    D8,
    [5.0, 6.0, 10.0, 11.0, 12.0, INF, 25.0, 27.0],
)
add_case(
    "misaligned-outer-x-inner-y",
    "synthetic-filtering-alignment",
    4,
    True,
    [
        ([dt.date(2020, 6, d) for d in (1, 2, 3, 4, 5, 6)], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ([dt.date(2020, 6, d) for d in (1, 3, 4, 5, 6, 7)], [2.0, 1.0, 4.0, 2.0, 5.0, 9.0]),
    ],
    [dt.date(2020, 6, d) for d in (1, 2, 3, 4, 5, 7)],
    [4.0, 99.0, 8.0, 13.0, 14.0, 88.0],
)
add_case(
    "leap-day-2020-02-29-intercept",
    "synthetic-leap",
    4,
    True,
    [(LEAP_DATES, [1.0, 2.0, 4.0, 5.0, 8.0, 10.0])],
    LEAP_DATES,
    [2.0, 5.0, 8.0, 11.0, 15.0, 21.0],
)
add_case(
    "leap-day-2020-02-29-no-intercept",
    "synthetic-leap",
    3,
    False,
    [(LEAP_DATES, [1.0, 2.0, 4.0, 5.0, 8.0, 10.0])],
    LEAP_DATES,
    [2.0, 5.0, 8.0, 11.0, 15.0, 21.0],
)
add_case(
    "negative-values-and-mixed-signs",
    "synthetic-filtering-alignment",
    4,
    True,
    [(D6, [-5.0, -2.0, 1.0, -3.0, 4.0, -1.0]), (D6, [3.0, -4.0, 2.0, 5.0, -1.0, 6.0])],
    D6,
    [-8.0, 3.0, -1.0, -10.0, 9.0, -7.0],
)
add_case(
    "perfect-fit-no-intercept",
    "synthetic-perfect-zero",
    4,
    False,
    [(D6, X6_QUAD), (D6, X6_ALT)],
    D6,
    [2.5 * a - 1.5 * b for a, b in zip(X6_QUAD, X6_ALT)],
)

# 6. Zero y and perfect fit with intercept (3 cases)
add_case(
    "zero-y-no-intercept",
    "synthetic-perfect-zero",
    3,
    False,
    [(D5, [1.0, 2.0, 4.0, 7.0, 11.0])],
    D5,
    [0.0] * 5,
)
add_case(
    "zero-y-with-intercept",
    "synthetic-perfect-zero",
    3,
    True,
    [(D5, [1.0, 2.0, 4.0, 7.0, 11.0])],
    D5,
    [0.0] * 5,
)
add_case(
    "perfect-fit-with-intercept",
    "synthetic-perfect-zero",
    4,
    True,
    [(D6, X6_QUAD), (D6, X6_ALT)],
    D6,
    [3.0 + 2.0 * a - 0.5 * b for a, b in zip(X6_QUAD, X6_ALT)],
)

# 7. Strict error cases (upstream MqValueError -> Error::Window) (4 cases)
add_case(
    "err-window-zero",
    "synthetic-errors",
    0,
    True,
    [(D6, X6_LIN)],
    D6,
    Y6_NOISY,
    rust_error="Window",
)
add_case(
    "err-window-eq-k-intercept",
    "synthetic-errors",
    3,
    True,
    [(D6, X6_QUAD), (D6, X6_ALT)],
    D6,
    Y6_NOISY,
    rust_error="Window",
)
add_case(
    "err-window-eq-k-no-intercept",
    "synthetic-errors",
    2,
    False,
    [(D6, X6_QUAD), (D6, X6_ALT)],
    D6,
    Y6_NOISY,
    rust_error="Window",
)
add_case(
    "err-window-eq-k-const-skip",
    "synthetic-errors",
    2,
    True,
    [(D6, X6_QUAD), (D6, [5.0] * 6)],
    D6,
    Y6_NOISY,
    rust_error="Window",
)

# 8. Documented parity boundaries (7 cases)
# 8a. Constant y with intercept (SST == 0): statsmodels normal-equation residual noise (~1e-29)
#     can emit r_squared = -Inf and std_err ~ 1e-15..1e-14; Rust returns exact r_squared = NaN and std_err = +0.0.
add_case(
    "doc-const-y-with-intercept",
    "documented-parity-boundaries",
    4,
    True,
    [(D6, [0.1, 0.3, 0.7, 1.3, 2.1, 3.1])],
    D6,
    [5.3] * 6,
    upstream_note="statsmodels_const_y_residual_noise",
)
# 8b. Ill-conditioned predictor (1e7 + [1..6]) with intercept: statsmodels normal equations lose ~5 digits,
#     whereas Rust Householder QR recovers accurate coefficients, r_squared, and std_err.
add_case(
    "doc-ill-conditioned-1e7",
    "documented-parity-boundaries",
    4,
    True,
    [(D6, [1e7 + k for k in range(1, 7)])],
    D6,
    [1.0 + 2.0 * k for k in range(1, 7)],
    upstream_note="statsmodels_normal_eq_ill_conditioned",
)
# 8c. Short or empty aligned series (aligned.len() < window): upstream ValueError / IndexError -> Error::Window.
add_case(
    "doc-short-aligned-lt-window",
    "documented-parity-boundaries",
    4,
    True,
    [(D6[:3], [1.0, 2.0, 4.0])],
    D6[:3],
    [3.0, 5.0, 9.0],
    rust_error="Window",
    upstream_note="upstream_short_aligned_error",
)
add_case(
    "doc-short-after-nan-filter",
    "documented-parity-boundaries",
    4,
    True,
    [(D6, [1.0, NAN, 3.0, INF, 5.0, 6.0])],
    D6,
    [2.0, 4.0, NINF, 8.0, 10.0, 12.0],
    rust_error="Window",
    upstream_note="upstream_short_aligned_error",
)
add_case(
    "doc-disjoint-dates-empty-aligned",
    "documented-parity-boundaries",
    4,
    True,
    [([dt.date(2020, 1, d) for d in range(1, 5)], [1.0, 2.0, 3.0, 4.0])],
    [dt.date(2020, 2, d) for d in range(1, 5)],
    [5.0, 6.0, 7.0, 8.0],
    rust_error="Window",
    upstream_note="upstream_short_aligned_error",
)
add_case(
    "doc-empty-series-aligned",
    "documented-parity-boundaries",
    4,
    True,
    [([], [])],
    [],
    [],
    rust_error="Window",
    upstream_note="upstream_short_aligned_error",
)
# 8d. Empty predictor list: upstream ValueError("No objects to concatenate") -> Error::Length.
add_case(
    "doc-empty-predictors",
    "documented-parity-boundaries",
    4,
    True,
    [],
    D6,
    Y6_NOISY,
    rust_error="Length",
    upstream_note="upstream_empty_predictors_value_error",
)

payload = {
    "schema": "gsquant-rolling-ols-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "statsmodels": sm.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/statistics.py:RollingLinearRegression",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "rolling-ols-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream RollingLinearRegression cases")
