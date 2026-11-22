"""Generate upstream cov parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries import Window
from gs_quant.timeseries.statistics import cov

ROOT = pathlib.Path(__file__).resolve().parent

UPSTREAM_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 3),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 7),
    dt.date(2019, 1, 8),
]
UPSTREAM_X = [3.0, 2.0, 3.0, 1.0, 3.0, 6.0]
UPSTREAM_Y = [3.5, 1.8, 2.9, 1.2, 3.1, 5.9]

D5 = [dt.date(2020, 1, d) for d in range(1, 6)]
X5 = [1.0, 2.0, 4.0, 7.0, 11.0]
Y5 = [2.0, 5.0, 3.0, 9.0, 12.0]
LIN5 = [1.0, 2.0, 3.0, 4.0, 5.0]

LEAP_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 5),
]
LEAP_X = [1.0, 3.0, 2.0, 6.0, 10.0]
LEAP_Y = [2.0, 5.0, 4.0, 11.0, 18.0]

PARTIAL_X_DATES = [dt.date(2020, 1, 1), dt.date(2020, 1, 3), dt.date(2020, 1, 5)]
PARTIAL_X_VALUES = [1.0, 3.0, 5.0]
PARTIAL_Y_DATES = D5
PARTIAL_Y_VALUES = [2.0, 10.0, 6.0, 20.0, 10.0]

INTER_X_DATES = [dt.date(2020, 1, d) for d in (1, 2, 4, 6)]
INTER_X_VALUES = [1.0, 4.0, 2.0, 8.0]
INTER_Y_DATES = [dt.date(2020, 1, d) for d in (1, 3, 4, 5, 6)]
INTER_Y_VALUES = [2.0, 7.0, 5.0, 9.0, 14.0]

DISJOINT_X_DATES = [dt.date(2020, 1, d) for d in (1, 2, 3)]
DISJOINT_X_VALUES = [1.0, 2.0, 3.0]
DISJOINT_Y_DATES = [dt.date(2020, 1, d) for d in (4, 5, 6)]
DISJOINT_Y_VALUES = [4.0, 5.0, 6.0]

SINGLE_DATES_1 = [dt.date(2020, 1, 1)]
SINGLE_DATES_2 = [dt.date(2020, 1, 2)]

LO_X = [1e9 + v for v in X5]
LO_Y = [1e9 + v for v in Y5]

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")


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
    args,
    x_dates,
    x_values,
    y_dates,
    y_values,
    fn,
    rust_error=None,
    upstream_note=None,
):
    sx = pd.Series(x_values, index=x_dates, dtype=float) if x_dates else pd.Series(dtype=float)
    sy = pd.Series(y_values, index=y_dates, dtype=float) if y_dates else pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "cov",
        "args": args,
        "input_x": pack_series(x_dates, x_values, sx),
        "input_y": pack_series(y_dates, y_values, sy),
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note
        if rust_error is not None:
            entry["rust_error"] = rust_error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = fn(sx, sy)
        except Exception as exc:
            if upstream_note == "upstream_empty_days_ramp_index_error":
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
            if upstream_note == "upstream_empty_days_ramp_index_error":
                raise AssertionError(f"{case_id}: expected upstream IndexError, got output")
            if rust_error is not None and upstream_note is None:
                raise AssertionError(
                    f"{case_id}: expected upstream error for {rust_error}, got output"
                )
            entry["output"] = {
                "index_kind": index_kind(out.index),
                "timezone": None,
                "dates": [fmt_date(d) for d in out.index],
                "values": [pack(float(v)) for v in out],
                "unit": "input_product",
                "dtype": str(out.dtype),
            }
    cases.append(entry)


# 1. Upstream test_statistics.py::test_cov suite (3 cases)
add_case(
    "upstream-exp",
    "test_statistics.py::test_cov",
    ["expanding", 0, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y),
)
add_case(
    "upstream-obs2",
    "test_statistics.py::test_cov",
    ["observations", 2, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(2, 0)),
)
add_case(
    "upstream-1w",
    "test_statistics.py::test_cov",
    ["days", 7, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window("1w", 0)),
)

# 2. Observation & calendar-day window/ramp variants, negative & self-covariance (11 cases)
add_case(
    "obs3-r0",
    "synthetic-windows",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "obs3-r2",
    "synthetic-windows",
    ["observations", 3, 2],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(3, 2)),
)
add_case(
    "obs1-r0",
    "synthetic-windows",
    ["observations", 1, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(1, 0)),
)
add_case(
    "obs-ramp-eq-len",
    "synthetic-windows",
    ["observations", 2, 6],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(2, 6)),
)
add_case(
    "obs-oversize",
    "synthetic-windows",
    ["observations", 10, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(10, 0)),
)
add_case(
    "days3-r0",
    "synthetic-windows",
    ["days", 3, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "days3-r3d",
    "synthetic-windows",
    ["days", 3, 3],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window("3d", "3d")),
)
add_case(
    "days1-r0",
    "synthetic-windows",
    ["days", 1, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window("1d", 0)),
)
add_case(
    "days-oversize-ramp",
    "synthetic-windows",
    ["days", 30, 30],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window("30d", "30d")),
)
add_case(
    "neg-cov-obs3",
    "synthetic-windows",
    ["observations", 3, 0],
    D5,
    LIN5,
    D5,
    [10.0, 8.0, 6.0, 4.0, 2.0],
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "self-cov-obs3",
    "synthetic-windows",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_X,
    lambda x, y: cov(x, y, Window(3, 0)),
)

# 3. Date alignment (partial, subset, superset, disjoint, interleaved) & leap day 2020-02-29 (13 cases)
add_case(
    "align-partial-exp",
    "synthetic-alignment",
    ["expanding", 0, 0],
    PARTIAL_X_DATES,
    PARTIAL_X_VALUES,
    PARTIAL_Y_DATES,
    PARTIAL_Y_VALUES,
    lambda x, y: cov(x, y),
)
add_case(
    "align-partial-obs3-r0",
    "synthetic-alignment",
    ["observations", 3, 0],
    INTER_X_DATES,
    INTER_X_VALUES,
    INTER_Y_DATES,
    INTER_Y_VALUES,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "align-partial-obs5-gt-lenx",
    "synthetic-alignment",
    ["observations", 5, 0],
    PARTIAL_X_DATES,
    PARTIAL_X_VALUES,
    PARTIAL_Y_DATES,
    PARTIAL_Y_VALUES,
    lambda x, y: cov(x, y, Window(5, 0)),
)
add_case(
    "align-partial-days4-r0",
    "synthetic-alignment",
    ["days", 4, 0],
    INTER_X_DATES,
    INTER_X_VALUES,
    INTER_Y_DATES,
    INTER_Y_VALUES,
    lambda x, y: cov(x, y, Window("4d", 0)),
)
add_case(
    "align-subset-x-in-y-obs3",
    "synthetic-alignment",
    ["observations", 3, 0],
    PARTIAL_X_DATES,
    PARTIAL_X_VALUES,
    PARTIAL_Y_DATES,
    PARTIAL_Y_VALUES,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "align-superset-x-of-y-days4",
    "synthetic-alignment",
    ["days", 4, 0],
    PARTIAL_Y_DATES,
    PARTIAL_Y_VALUES,
    PARTIAL_X_DATES,
    PARTIAL_X_VALUES,
    lambda x, y: cov(x, y, Window("4d", 0)),
)
add_case(
    "align-disjoint-obs3",
    "synthetic-alignment",
    ["observations", 3, 0],
    DISJOINT_X_DATES,
    DISJOINT_X_VALUES,
    DISJOINT_Y_DATES,
    DISJOINT_Y_VALUES,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "align-disjoint-days3",
    "synthetic-alignment",
    ["days", 3, 0],
    DISJOINT_X_DATES,
    DISJOINT_X_VALUES,
    DISJOINT_Y_DATES,
    DISJOINT_Y_VALUES,
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "align-interleaved-obs4-r1",
    "synthetic-alignment",
    ["observations", 4, 1],
    INTER_X_DATES,
    INTER_X_VALUES,
    INTER_Y_DATES,
    INTER_Y_VALUES,
    lambda x, y: cov(x, y, Window(4, 1)),
)
add_case(
    "align-interleaved-days4-r2d",
    "synthetic-alignment",
    ["days", 4, 2],
    INTER_X_DATES,
    INTER_X_VALUES,
    INTER_Y_DATES,
    INTER_Y_VALUES,
    lambda x, y: cov(x, y, Window("4d", "2d")),
)
add_case(
    "leap-days3-r0",
    "synthetic-leap",
    ["days", 3, 0],
    LEAP_DATES,
    LEAP_X,
    LEAP_DATES,
    LEAP_Y,
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "leap-days3-r3d",
    "synthetic-leap",
    ["days", 3, 3],
    LEAP_DATES,
    LEAP_X,
    LEAP_DATES,
    LEAP_Y,
    lambda x, y: cov(x, y, Window("3d", "3d")),
)
add_case(
    "leap-days7-r0",
    "synthetic-leap",
    ["days", 7, 0],
    LEAP_DATES,
    LEAP_X,
    LEAP_DATES,
    LEAP_Y,
    lambda x, y: cov(x, y, Window("7d", 0)),
)

# 4. Empty, singleton, asymmetric empty, and strict error boundaries (22 cases)
add_case(
    "empty-both-exp",
    "synthetic-boundaries",
    ["expanding", 0, 0],
    [],
    [],
    [],
    [],
    lambda x, y: cov(x, y),
)
add_case(
    "empty-both-obs2",
    "synthetic-boundaries",
    ["observations", 2, 0],
    [],
    [],
    [],
    [],
    lambda x, y: cov(x, y, Window(2, 0)),
)
add_case(
    "empty-both-obs0",
    "synthetic-boundaries",
    ["observations", 0, 0],
    [],
    [],
    [],
    [],
    lambda x, y: cov(x, y, Window(0, 0)),
)
add_case(
    "empty-both-obs2-r5",
    "synthetic-boundaries",
    ["observations", 2, 5],
    [],
    [],
    [],
    [],
    lambda x, y: cov(x, y, Window(2, 5)),
)
add_case(
    "empty-both-days3-r0",
    "synthetic-boundaries",
    ["days", 3, 0],
    [],
    [],
    [],
    [],
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "empty-both-days0-r0",
    "synthetic-boundaries",
    ["days", 0, 0],
    [],
    [],
    [],
    [],
    lambda x, y: cov(x, y, Window("0d", 0)),
)
add_case(
    "empty-x-nonempty-y-obs2",
    "synthetic-boundaries",
    ["observations", 2, 0],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y, Window(2, 0)),
)
add_case(
    "empty-x-nonempty-y-obs2-r2",
    "synthetic-boundaries",
    ["observations", 2, 2],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y, Window(2, 2)),
)
add_case(
    "empty-x-nonempty-y-obs10",
    "synthetic-boundaries",
    ["observations", 10, 0],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y, Window(10, 0)),
)
add_case(
    "empty-x-nonempty-y-days3-r0",
    "synthetic-boundaries",
    ["days", 3, 0],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "empty-x-nonempty-y-exp-err",
    "synthetic-boundaries",
    ["expanding", 0, 0],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y),
    rust_error="Window",
)
add_case(
    "empty-x-nonempty-y-obs0-err",
    "synthetic-boundaries",
    ["observations", 0, 0],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y, Window(0, 0)),
    rust_error="Window",
)
add_case(
    "empty-x-nonempty-y-ramp-gt-y-err",
    "synthetic-boundaries",
    ["observations", 2, 6],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y, Window(2, 6)),
    rust_error="Ramp",
)
add_case(
    "nonempty-x-empty-y-exp",
    "synthetic-boundaries",
    ["expanding", 0, 0],
    D5,
    X5,
    [],
    [],
    lambda x, y: cov(x, y),
)
add_case(
    "nonempty-x-empty-y-days3-r0",
    "synthetic-boundaries",
    ["days", 3, 0],
    D5,
    X5,
    [],
    [],
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "singleton-same-date-exp",
    "synthetic-boundaries",
    ["expanding", 0, 0],
    SINGLE_DATES_1,
    [5.0],
    SINGLE_DATES_1,
    [7.0],
    lambda x, y: cov(x, y),
)
add_case(
    "singleton-diff-dates-obs2",
    "synthetic-boundaries",
    ["observations", 2, 0],
    SINGLE_DATES_1,
    [5.0],
    SINGLE_DATES_2,
    [7.0],
    lambda x, y: cov(x, y, Window(2, 0)),
)
add_case(
    "err-obs-size0",
    "synthetic-boundaries",
    ["observations", 0, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(0, 0)),
    rust_error="Window",
)
add_case(
    "err-obs-size0-ramp-gt-len",
    "synthetic-boundaries",
    ["observations", 0, 10],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(0, 10)),
    rust_error="Window",
)
add_case(
    "err-obs-ramp-gt-len",
    "synthetic-boundaries",
    ["observations", 2, 7],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(2, 7)),
    rust_error="Ramp",
)
add_case(
    "err-obs-oversize-ramp-gt-len",
    "synthetic-boundaries",
    ["observations", 10, 10],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window(10, 10)),
    rust_error="Ramp",
)
add_case(
    "err-partial-ramp-gt-lenx",
    "synthetic-boundaries",
    ["observations", 4, 4],
    PARTIAL_X_DATES,
    PARTIAL_X_VALUES,
    PARTIAL_Y_DATES,
    PARTIAL_Y_VALUES,
    lambda x, y: cov(x, y, Window(4, 4)),
    rust_error="Ramp",
)

# 5. NaN, ±Inf, exact constants, signed zeros, orthogonal zero, large-offset days, overflow (16 cases)
add_case(
    "nan-mid-obs3",
    "synthetic-nonfinite",
    ["observations", 3, 0],
    D5,
    [1.0, NAN, 3.0, 4.0, 5.0],
    D5,
    [2.0, 4.0, NAN, 8.0, 10.0],
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "nan-mid-days4",
    "synthetic-nonfinite",
    ["days", 4, 0],
    D5,
    [1.0, NAN, 3.0, 4.0, 5.0],
    D5,
    [2.0, 4.0, NAN, 8.0, 10.0],
    lambda x, y: cov(x, y, Window("4d", 0)),
)
add_case(
    "all-nan-obs3",
    "synthetic-nonfinite",
    ["observations", 3, 0],
    D5,
    [NAN] * 5,
    D5,
    LIN5,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "inf-obs3-skip",
    "synthetic-nonfinite",
    ["observations", 3, 0],
    D5,
    [1.0, INF, 3.0, 4.0, 5.0],
    D5,
    [2.0, 4.0, 6.0, 8.0, 10.0],
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "inf-exp-skip",
    "synthetic-nonfinite",
    ["expanding", 0, 0],
    D5,
    [1.0, INF, 3.0, NINF, 5.0],
    D5,
    [2.0, 4.0, 6.0, 8.0, 10.0],
    lambda x, y: cov(x, y),
)
add_case(
    "inf-days3-finite-pair",
    "synthetic-nonfinite",
    ["days", 3, 0],
    D5,
    [1.0, INF, 3.0, 4.0, 5.0],
    D5,
    [2.0, 4.0, 6.0, 8.0, 10.0],
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "inf-days3-nan-pair",
    "synthetic-nonfinite",
    ["days", 3, 0],
    D5,
    [1.0, INF, 3.0, 4.0, 5.0],
    D5,
    [2.0, NAN, 6.0, 8.0, 10.0],
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "inf-days3-unmatched",
    "synthetic-nonfinite",
    ["days", 3, 0],
    D5,
    [1.0, INF, 3.0, 4.0, 5.0],
    [D5[0], D5[2], D5[3], D5[4]],
    [2.0, 6.0, 8.0, 10.0],
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "const-exact-obs3",
    "synthetic-zeros",
    ["observations", 3, 0],
    D5,
    [2.0] * 5,
    D5,
    LIN5,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "const-neg0-obs3",
    "synthetic-zeros",
    ["observations", 3, 0],
    D5,
    [-0.0] * 5,
    D5,
    [-0.0] * 5,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "const-neg0-days3",
    "synthetic-zeros",
    ["days", 3, 0],
    D5,
    [-0.0] * 5,
    D5,
    LIN5,
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "mixed-signed-zeros-obs3",
    "synthetic-zeros",
    ["observations", 3, 0],
    D5,
    [-0.0, 0.0, -0.0, 0.0, -0.0],
    D5,
    LIN5,
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "ortho-zero-obs3",
    "synthetic-zeros",
    ["observations", 3, 0],
    D5[:4],
    [1.0, 2.0, 1.0, 2.0],
    D5[:4],
    [1.0, 2.0, 3.0, 4.0],
    lambda x, y: cov(x, y, Window(3, 0)),
)
add_case(
    "large-offset-days3",
    "synthetic-stability",
    ["days", 3, 0],
    D5,
    LO_X,
    D5,
    LO_Y,
    lambda x, y: cov(x, y, Window("3d", 0)),
)
add_case(
    "overflow-pos-obs2",
    "synthetic-overflow",
    ["observations", 2, 0],
    D5[:2],
    [1e308, -1e308],
    D5[:2],
    [1e308, -1e308],
    lambda x, y: cov(x, y, Window(2, 0)),
)
add_case(
    "overflow-neg-days2",
    "synthetic-overflow",
    ["days", 2, 0],
    D5[:2],
    [1e308, -1e308],
    D5[:2],
    [-1e308, 1e308],
    lambda x, y: cov(x, y, Window("2d", 0)),
)

# 6. Documented parity boundaries (7 cases covering all 4 runtime documented diffs)
# 6a. Window::Days { size: 0, .. } on non-empty x: upstream DateOffset(days=0) yields all NaN;
#     Rust consistently returns Error::Window.
add_case(
    "err-days0",
    "documented-parity-boundaries",
    ["days", 0, 0],
    UPSTREAM_DATES,
    UPSTREAM_X,
    UPSTREAM_DATES,
    UPSTREAM_Y,
    lambda x, y: cov(x, y, Window("0d", 0)),
    rust_error="Window",
    upstream_note="upstream_zero_day_offset_all_nan",
)
# 6b. Empty x with Window::Days { size > 0, ramp > 0 }: upstream apply_ramp raises IndexError;
#     Rust returns empty Series.
add_case(
    "empty-both-days3-r1d",
    "documented-parity-boundaries",
    ["days", 3, 1],
    [],
    [],
    [],
    [],
    lambda x, y: cov(x, y, Window("3d", "1d")),
    upstream_note="upstream_empty_days_ramp_index_error",
)
add_case(
    "empty-x-nonempty-y-days3-r1d",
    "documented-parity-boundaries",
    ["days", 3, 1],
    [],
    [],
    D5,
    Y5,
    lambda x, y: cov(x, y, Window("3d", "1d")),
    upstream_note="upstream_empty_days_ramp_index_error",
)
# 6c. Constant finite windows: upstream pandas/NumPy leave floating-point rounding residue
#     (-2.66e-15 on 4.2 in rolling().cov(); 2.89e-34 on 0.1 in np.cov); Rust returns exact +0.0.
add_case(
    "const-42-obs3",
    "documented-parity-boundaries",
    ["observations", 3, 0],
    D5,
    [4.2] * 5,
    D5,
    LIN5,
    lambda x, y: cov(x, y, Window(3, 0)),
    upstream_note="upstream_constant_window_rounding_residue",
)
add_case(
    "const-01-days3",
    "documented-parity-boundaries",
    ["days", 3, 0],
    D5,
    [0.1] * 5,
    D5,
    [0.1] * 5,
    lambda x, y: cov(x, y, Window("3d", 0)),
    upstream_note="upstream_constant_window_rounding_residue",
)
# 6d. Large-offset observation/expanding windows: pinned pandas 2.3.3 rolling().cov() suffers
#     catastrophic cancellation from single-pass E[XY] - E[X]E[Y]; Rust two-pass centered
#     covariance matches exact mathematical covariance (and upstream Window::Days / np.cov).
add_case(
    "large-offset-obs3",
    "documented-parity-boundaries",
    ["observations", 3, 0],
    D5,
    LO_X,
    D5,
    LO_Y,
    lambda x, y: cov(x, y, Window(3, 0)),
    upstream_note="pandas_rolling_cov_large_offset_cancellation",
)
add_case(
    "large-offset-exp",
    "documented-parity-boundaries",
    ["expanding", 0, 0],
    D5,
    LO_X,
    D5,
    LO_Y,
    lambda x, y: cov(x, y),
    upstream_note="pandas_rolling_cov_large_offset_cancellation",
)

payload = {
    "schema": "gsquant-cov-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/statistics.py:cov",
        "gs_quant/timeseries/helper.py:normalize_window",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "cov-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream cov cases")
