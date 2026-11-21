"""Generate upstream beta parity vectors only. Run with the locked upstream Python environment."""
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import warnings

# The caller supplies the fixed checkout; no network or Rust execution here.
sys.path.insert(0, sys.argv[1])
import pandas as pd
from gs_quant.timeseries import Window, beta, returns

ROOT = pathlib.Path(__file__).resolve().parent

DAILY_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 3),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 7),
    dt.date(2019, 1, 8),
]
TEST_X_VALUES = [100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792]
TEST_Y_VALUES = [100.0, 102.0, 104.0, 101.0, 100.95, 100.0]

ASSET_DATES = [dt.date(2020, 1, d) for d in (1, 2, 4, 5, 7, 10, 11, 13)]
ASSET_VALUES = [1.0, 4.0, 2.0, 9.0, 3.0, 5.0, 6.0, 2.0]
BENCH_DATES = [dt.date(2020, 1, d) for d in (1, 3, 4, 5, 7, 8, 10, 11, 13)]
BENCH_VALUES = [3.0, 5.0, 1.0, 9.0, 2.0, 4.0, 8.0, 3.0, 7.0]
MISSING_BENCH_VALUES = [3.0, 5.0, 1.0, 9.0, float("nan"), 4.0, 8.0, 3.0, 7.0]
CONST_BENCH_VALUES = [0.1] * 8
DISJOINT_DATES = [dt.date(2020, 1, d) for d in (14, 15, 16, 17)]
DISJOINT_VALUES = [2.0, 3.0, 1.0, 4.0]
ASSET_PRICE_VALUES = [v + 100.0 for v in ASSET_VALUES]
BENCH_PRICE_VALUES = [v + 100.0 for v in BENCH_VALUES]


def pack(v: float):
    if math.isnan(v):
        return {"special": "NaN"}
    if math.isinf(v):
        return {"special": "+Inf" if v > 0 else "-Inf"}
    return float(v)


def pack_series(dates, values, series):
    return {
        "index_kind": "Index[date]" if dates else type(series.index).__name__,
        "timezone": None,
        "dates": [str(d) for d in dates],
        "values": [pack(v) for v in values],
        "dtype": "float64",
    }


cases = []


def add_case(
    case_id,
    source_ref,
    args,
    x_dates,
    x_values,
    b_dates,
    b_values,
    fn,
    rust_error=None,
    upstream_note=None,
):
    sx = pd.Series(x_values, index=x_dates, dtype=float) if x_dates else pd.Series(dtype=float)
    sb = pd.Series(b_values, index=b_dates, dtype=float) if b_dates else pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "beta",
        "args": args,
        "input_x": pack_series(x_dates, x_values, sx),
        "input_b": pack_series(b_dates, b_values, sb),
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = fn(sx, sb)
        except Exception as exc:
            if rust_error is None:
                raise
            entry["error"] = {
                "rust": rust_error,
                "upstream_type": type(exc).__name__,
                "upstream_message": str(exc),
            }
        else:
            if rust_error is not None:
                raise AssertionError(
                    f"{case_id}: expected upstream error for {rust_error}, got output"
                )
            entry["output"] = {
                "index_kind": "Index[date]" if len(out) else type(out.index).__name__,
                "timezone": None,
                "dates": [str(d) for d in out.index],
                "values": [pack(float(v)) for v in out],
                "unit": "ratio",
                "dtype": "float64",
            }
    cases.append(entry)


# 8 representative cases from upstream test_econometrics.py::test_beta
_ret_x = returns(pd.Series(TEST_X_VALUES, index=DAILY_DATES, dtype=float))
_ret_y = returns(pd.Series(TEST_Y_VALUES, index=DAILY_DATES, dtype=float))
TEST_RET_X_VALUES = [float(v) for v in _ret_x.values]
TEST_RET_Y_VALUES = [float(v) for v in _ret_y.values]

add_case(
    "test-beta-empty-default",
    "test_econometrics.py::test_beta",
    ["expanding", 0, 0, "true"],
    [],
    [],
    [],
    [],
    lambda x, b: beta(x, b),
)
add_case(
    "test-beta-empty-obs-1",
    "test_econometrics.py::test_beta",
    ["observations", 1, 1, "true"],
    [],
    [],
    [],
    [],
    lambda x, b: beta(x, b, 1),
)
add_case(
    "test-beta-prices-identical-default",
    "test_econometrics.py::test_beta",
    ["expanding", 0, 0, "true"],
    DAILY_DATES,
    TEST_X_VALUES,
    DAILY_DATES,
    TEST_X_VALUES,
    lambda x, b: beta(x, b),
)
add_case(
    "test-beta-prices-default",
    "test_econometrics.py::test_beta",
    ["expanding", 0, 0, "true"],
    DAILY_DATES,
    TEST_X_VALUES,
    DAILY_DATES,
    TEST_Y_VALUES,
    lambda x, b: beta(x, b),
)
add_case(
    "test-beta-prices-obs-2",
    "test_econometrics.py::test_beta",
    ["observations", 2, 0, "true"],
    DAILY_DATES,
    TEST_X_VALUES,
    DAILY_DATES,
    TEST_Y_VALUES,
    lambda x, b: beta(x, b, Window(2, 0)),
)
add_case(
    "test-beta-returns-obs-2",
    "test_econometrics.py::test_beta",
    ["observations", 2, 0, "false"],
    DAILY_DATES,
    TEST_RET_X_VALUES,
    DAILY_DATES,
    TEST_RET_Y_VALUES,
    lambda x, b: beta(x, b, Window(2, 0), False),
)
add_case(
    "test-beta-prices-2d-ramp-0",
    "test_econometrics.py::test_beta",
    ["days", 2, 0, "true"],
    DAILY_DATES,
    TEST_X_VALUES,
    DAILY_DATES,
    TEST_Y_VALUES,
    lambda x, b: beta(x, b, Window("2d", 0)),
)
add_case(
    "test-beta-prices-2d-default-ramp",
    "test_econometrics.py::test_beta",
    ["days", 2, 2, "true"],
    DAILY_DATES,
    TEST_X_VALUES,
    DAILY_DATES,
    TEST_Y_VALUES,
    lambda x, b: beta(x, b, "2d"),
)

# 13 bundled synthetic boundary cases from boundaries.py / boundaries-1.json
add_case(
    "observations-offset-dates",
    "boundaries-1.json",
    ["observations", 4, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    BENCH_VALUES,
    lambda x, b: beta(x, b, Window(4, 0), False),
)
add_case(
    "calendar-offset-dates",
    "boundaries-1.json",
    ["days", 5, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    BENCH_VALUES,
    lambda x, b: beta(x, b, Window("5d", 0), False),
)
add_case(
    "calendar-both-default-ramps",
    "boundaries-1.json",
    ["days", 5, 5, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    BENCH_VALUES,
    lambda x, b: beta(x, b, "5d", False),
)
add_case(
    "observations-missing-benchmark",
    "boundaries-1.json",
    ["observations", 4, 2, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    MISSING_BENCH_VALUES,
    lambda x, b: beta(x, b, Window(4, 2), False),
)
add_case(
    "calendar-missing-benchmark",
    "boundaries-1.json",
    ["days", 5, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    MISSING_BENCH_VALUES,
    lambda x, b: beta(x, b, Window("5d", 0), False),
)
add_case(
    "prices",
    "boundaries-1.json",
    ["observations", 5, 0, "true"],
    ASSET_DATES,
    ASSET_PRICE_VALUES,
    BENCH_DATES,
    BENCH_PRICE_VALUES,
    lambda x, b: beta(x, b, Window(5, 0), True),
)
add_case(
    "constant-observations",
    "boundaries-1.json",
    ["observations", 4, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    ASSET_DATES,
    CONST_BENCH_VALUES,
    lambda x, b: beta(x, b, Window(4, 0), False),
)
add_case(
    "constant-calendar",
    "boundaries-1.json",
    ["days", 6, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    ASSET_DATES,
    CONST_BENCH_VALUES,
    lambda x, b: beta(x, b, Window("6d", 0), False),
    upstream_note="numpy_calendar_const_var_rounding_residue",
)
add_case(
    "disjoint-observations",
    "boundaries-1.json",
    ["observations", 4, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    DISJOINT_DATES,
    DISJOINT_VALUES,
    lambda x, b: beta(x, b, Window(4, 0), False),
)
add_case(
    "disjoint-calendar",
    "boundaries-1.json",
    ["days", 6, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    DISJOINT_DATES,
    DISJOINT_VALUES,
    lambda x, b: beta(x, b, Window("6d", 0), False),
)
add_case(
    "oversize",
    "boundaries-1.json",
    ["observations", 12, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    BENCH_VALUES,
    lambda x, b: beta(x, b, Window(12, 0), False),
)
add_case(
    "oversize-invalid-ramp",
    "boundaries-1.json",
    ["observations", 12, 12, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    BENCH_VALUES,
    lambda x, b: beta(x, b, Window(12, 12), False),
    rust_error="Ramp",
)
add_case(
    "empty",
    "boundaries-1.json",
    ["expanding", 0, 0, "false"],
    [],
    [],
    [],
    [],
    lambda x, b: beta(x, b, Window(None, 0), False),
)

# 2 additional boundary cases: expanding on offset dates and invalid zero window
add_case(
    "expanding-offset-dates",
    "boundaries-extra",
    ["expanding", 0, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    BENCH_VALUES,
    lambda x, b: beta(x, b, Window(None, 0), False),
)
add_case(
    "invalid-window-zero",
    "boundaries-extra",
    ["observations", 0, 0, "false"],
    ASSET_DATES,
    ASSET_VALUES,
    BENCH_DATES,
    BENCH_VALUES,
    lambda x, b: beta(x, b, Window(0, 0), False),
    rust_error="Window",
)

payload = {
    "schema": "gsquant-beta-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/econometrics.py:beta",
        "gs_quant/timeseries/econometrics.py:returns",
        "gs_quant/timeseries/datetime.py:interpolate",
        "gs_quant/timeseries/helper.py:normalize_window",
        "gs_quant/timeseries/helper.py:apply_ramp",
        "gs_quant/timeseries/helper.py:window_cutoff",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "beta-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream beta cases")
