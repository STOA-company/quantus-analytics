"""Generate upstream vol_swap_volatility and corr_swap_correlation parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries.econometrics import corr_swap_correlation, vol_swap_volatility

ROOT = pathlib.Path(__file__).resolve().parent

# Upstream test_vol_swap_volatility deterministic 50-day series (seed=100)
_rng_vol = np.random.default_rng(100)
UP_VOL_DATES = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(50)]
_up_vol_rets = _rng_vol.normal(0.0, 0.01, 49)
UP_VOL_VALUES = [
    float(v) for v in (100.0 * np.exp(np.cumsum(np.concatenate([[0.0], _up_vol_rets]))))
]

# Upstream test_corr_swap_correlation deterministic 10-day series (fixed seed=42)
_rng_corr = np.random.default_rng(42)
UP_CORR_DATES = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(10)]
UP_CORR_X_VALUES = [float(v) for v in _rng_corr.random(10)]
UP_CORR_Y_VALUES = [float(v) for v in _rng_corr.random(10)]

# Synthetic 6-day series
D6 = [dt.date(2024, 1, i) for i in range(1, 7)]
P6 = [100.0, 102.0, 101.0, 104.0, 103.0, 106.0]
Q6 = [50.0, 51.5, 50.2, 52.0, 52.5, 54.0]
CONST6 = [100.0] * 6
CONST_LOGRET6 = [100.0 * (1.25**i) for i in range(6)]
NAN_P6 = [100.0, 102.0, float("nan"), 104.0, 103.0, 106.0]
NAN_Q6 = [50.0, 51.5, 50.2, float("nan"), 52.5, 54.0]
D8 = [dt.date(2024, 1, i) for i in range(1, 9)]
NAN_P8 = [100.0, 102.0, 101.0, float("nan"), 104.0, 103.0, 106.0, 105.0]
NAN_Q8 = [50.0, 51.5, 50.2, 52.0, 52.5, 54.0, 53.5, 55.0]

# Leap-year 2020-02-29 series
LEAP_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 1),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 3),
]
LEAP_X = [100.0, 102.0, 101.0, 104.0, 103.0, 106.0]
LEAP_Y = [50.0, 51.5, 50.2, 52.0, 52.5, 54.0]

# Irregular 3-day spacing series
IRREG_DATES = [dt.date(2024, 1, 1) + dt.timedelta(days=3 * i) for i in range(5)]
IRREG_X = [100.0, 102.0, 101.0, 104.0, 103.0]
IRREG_Y = [50.0, 51.0, 50.5, 52.0, 51.5]

# Misaligned calendar series
MIS_X_DATES = [dt.date(2024, 3, d) for d in (1, 2, 3, 4, 5, 6)]
MIS_X_VALUES = [100.0, 102.0, 104.0, 103.0, 106.0, 108.0]
MIS_Y_DATES = [dt.date(2024, 3, d) for d in (1, 3, 4, 5, 6)]
MIS_Y_VALUES = [200.0, 206.0, 204.0, 210.0, 215.0]

SINGLE_DATES = [dt.date(2024, 1, 1)]
SINGLE_VALUES = [100.0]

TWO_DATES = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
TWO_X = [100.0, 102.0]
TWO_Y = [50.0, 51.0]

NONPOS_ZERO_DATES = D6[:4]
NONPOS_ZERO_VALUES = [100.0, 0.0, 102.0, 103.0]
NONPOS_NEG_VALUES = [100.0, -5.0, 102.0, 103.0]


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


def add_vol_case(case_id, source_ref, args, dates, values, fn, rust_error=None):
    series = pd.Series(values, index=dates, dtype=float) if dates else pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "vol_swap_volatility",
        "args": args,
        "input": pack_series(dates, values, series),
    }
    try:
        out = fn(series)
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
            raise AssertionError(f"{case_id}: expected upstream error for {rust_error}, got output")
        entry["output"] = {
            "index_kind": "Index[date]" if len(out) else type(out.index).__name__,
            "timezone": None,
            "dates": [str(d) for d in out.index],
            "values": [pack(float(v)) for v in out],
            "unit": "percent",
            "dtype": "float64",
        }
    cases.append(entry)


def add_corr_case(
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
        "operation": "corr_swap_correlation",
        "args": args,
        "input_x": pack_series(x_dates, x_values, sx),
        "input_y": pack_series(y_dates, y_values, sy),
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = fn(sx, sy)
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
            raise AssertionError(f"{case_id}: expected upstream error for {rust_error}, got output")
        entry["output"] = {
            "index_kind": "Index[date]" if len(out) else type(out.index).__name__,
            "timezone": None,
            "dates": [str(d) for d in out.index],
            "values": [pack(float(v)) for v in out],
            "unit": "ratio",
            "dtype": "float64",
        }
    cases.append(entry)


# =============================================================================
# vol_swap_volatility cases (33 cases)
# =============================================================================

# Upstream test_vol_swap_volatility representative suite (6)
add_vol_case(
    "vol-upstream-n3-zm-true",
    "test_vol_swap_volatility",
    [3, "default", "true"],
    UP_VOL_DATES,
    UP_VOL_VALUES,
    lambda x: vol_swap_volatility(x, n_days=3),
)
add_vol_case(
    "vol-upstream-n22-default",
    "test_vol_swap_volatility",
    [22, "default", "true"],
    UP_VOL_DATES,
    UP_VOL_VALUES,
    lambda x: vol_swap_volatility(x, n_days=22),
)
add_vol_case(
    "vol-upstream-n22-af365",
    "test_vol_swap_volatility",
    [22, 365, "true"],
    UP_VOL_DATES,
    UP_VOL_VALUES,
    lambda x: vol_swap_volatility(x, n_days=22, annualization_factor=365),
)
add_vol_case(
    "vol-upstream-n22-zm-false",
    "test_vol_swap_volatility",
    [22, "default", "false"],
    UP_VOL_DATES,
    UP_VOL_VALUES,
    lambda x: vol_swap_volatility(x, n_days=22, assume_zero_mean=False),
)
add_vol_case(
    "vol-upstream-default-full",
    "test_vol_swap_volatility",
    ["default", "default", "true"],
    UP_VOL_DATES,
    UP_VOL_VALUES,
    lambda x: vol_swap_volatility(x),
)
add_vol_case(
    "vol-upstream-default-full-zm-false",
    "test_vol_swap_volatility",
    ["default", "default", "false"],
    UP_VOL_DATES,
    UP_VOL_VALUES,
    lambda x: vol_swap_volatility(x, assume_zero_mean=False),
)

# Custom annualization factors (4)
add_vol_case(
    "vol-af-0-zm-true",
    "synthetic-factors",
    [3, 0, "true"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=3, annualization_factor=0, assume_zero_mean=True),
)
add_vol_case(
    "vol-af-0-zm-false",
    "synthetic-factors",
    [3, 0, "false"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=3, annualization_factor=0, assume_zero_mean=False),
)
add_vol_case(
    "vol-af-52-zm-true",
    "synthetic-factors",
    [4, 52, "true"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=4, annualization_factor=52, assume_zero_mean=True),
)
add_vol_case(
    "vol-af-255-zm-false",
    "synthetic-factors",
    [4, 255, "false"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=4, annualization_factor=255, assume_zero_mean=False),
)

# Leap day 2020-02-29 & irregular 3-day spacing (3)
add_vol_case(
    "vol-leap-2020-n4-zm-true",
    "synthetic-calendar",
    [4, "default", "true"],
    LEAP_DATES,
    LEAP_X,
    lambda x: vol_swap_volatility(x, n_days=4, assume_zero_mean=True),
)
add_vol_case(
    "vol-leap-2020-n4-zm-false",
    "synthetic-calendar",
    [4, "default", "false"],
    LEAP_DATES,
    LEAP_X,
    lambda x: vol_swap_volatility(x, n_days=4, assume_zero_mean=False),
)
add_vol_case(
    "vol-irregular-3d-n3-zm-true",
    "synthetic-calendar",
    [3, "default", "true"],
    IRREG_DATES,
    IRREG_X,
    lambda x: vol_swap_volatility(x, n_days=3, assume_zero_mean=True),
)

# Window & ramp boundaries (7)
add_vol_case(
    "vol-n1-zm-true",
    "synthetic-boundaries",
    [1, "default", "true"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=1, assume_zero_mean=True),
)
add_vol_case(
    "vol-n1-zm-false",
    "synthetic-boundaries",
    [1, "default", "false"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=1, assume_zero_mean=False),
)
add_vol_case(
    "vol-n2-zm-true",
    "synthetic-boundaries",
    [2, "default", "true"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=2, assume_zero_mean=True),
)
add_vol_case(
    "vol-n2-zm-false",
    "synthetic-boundaries",
    [2, "default", "false"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=2, assume_zero_mean=False),
)
add_vol_case(
    "vol-oversize-len-plus-1",
    "synthetic-boundaries",
    [7, "default", "true"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=7, assume_zero_mean=True),
)
add_vol_case(
    "vol-oversize-len-plus-2-ramp-err",
    "synthetic-boundaries",
    [8, "default", "true"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=8, assume_zero_mean=True),
    rust_error="Ramp",
)
add_vol_case(
    "vol-n0-window-err",
    "synthetic-boundaries",
    [0, "default", "true"],
    D6,
    P6,
    lambda x: vol_swap_volatility(x, n_days=0, assume_zero_mean=True),
    rust_error="Window",
)

# Empty, singleton, two-point (7)
add_vol_case(
    "vol-empty-default",
    "synthetic-shapes",
    ["default", "default", "true"],
    [],
    [],
    lambda x: vol_swap_volatility(x),
)
add_vol_case(
    "vol-empty-n0",
    "synthetic-shapes",
    [0, "default", "true"],
    [],
    [],
    lambda x: vol_swap_volatility(x, n_days=0),
)
add_vol_case(
    "vol-empty-n5",
    "synthetic-shapes",
    [5, "default", "false"],
    [],
    [],
    lambda x: vol_swap_volatility(x, n_days=5, assume_zero_mean=False),
)
add_vol_case(
    "vol-singleton-default-zm-true",
    "synthetic-shapes",
    ["default", "default", "true"],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: vol_swap_volatility(x, assume_zero_mean=True),
)
add_vol_case(
    "vol-singleton-default-zm-false",
    "synthetic-shapes",
    ["default", "default", "false"],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: vol_swap_volatility(x, assume_zero_mean=False),
)
add_vol_case(
    "vol-two-point-default-zm-true",
    "synthetic-shapes",
    ["default", "default", "true"],
    TWO_DATES,
    TWO_X,
    lambda x: vol_swap_volatility(x, assume_zero_mean=True),
)
add_vol_case(
    "vol-two-point-default-zm-false",
    "synthetic-shapes",
    ["default", "default", "false"],
    TWO_DATES,
    TWO_X,
    lambda x: vol_swap_volatility(x, assume_zero_mean=False),
)

# Constant, NaN, non-positive errors (6)
add_vol_case(
    "vol-constant-n3-zm-true",
    "synthetic-values",
    [3, "default", "true"],
    D6,
    CONST6,
    lambda x: vol_swap_volatility(x, n_days=3, assume_zero_mean=True),
)
add_vol_case(
    "vol-constant-n3-zm-false",
    "synthetic-values",
    [3, "default", "false"],
    D6,
    CONST6,
    lambda x: vol_swap_volatility(x, n_days=3, assume_zero_mean=False),
)
add_vol_case(
    "vol-nan-mid-n3-zm-true",
    "synthetic-values",
    [3, "default", "true"],
    D6,
    NAN_P6,
    lambda x: vol_swap_volatility(x, n_days=3, assume_zero_mean=True),
)
add_vol_case(
    "vol-nan-mid-n3-zm-false",
    "synthetic-values",
    [3, "default", "false"],
    D6,
    NAN_P6,
    lambda x: vol_swap_volatility(x, n_days=3, assume_zero_mean=False),
)
add_vol_case(
    "vol-nonpositive-zero-err",
    "synthetic-errors",
    [3, "default", "true"],
    NONPOS_ZERO_DATES,
    NONPOS_ZERO_VALUES,
    lambda x: vol_swap_volatility(x, n_days=3, assume_zero_mean=True),
    rust_error="NonPositivePrice",
)
add_vol_case(
    "vol-nonpositive-neg-err",
    "synthetic-errors",
    [3, "default", "false"],
    NONPOS_ZERO_DATES,
    NONPOS_NEG_VALUES,
    lambda x: vol_swap_volatility(x, n_days=3, assume_zero_mean=False),
    rust_error="NonPositivePrice",
)


# =============================================================================
# corr_swap_correlation cases (42 cases)
# =============================================================================

# Upstream test_corr_swap_correlation representative suite (8)
add_corr_case(
    "corr-upstream-default-zm-true",
    "test_corr_swap_correlation",
    ["default", "true"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y),
)
add_corr_case(
    "corr-upstream-default-zm-false",
    "test_corr_swap_correlation",
    ["default", "false"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, assume_zero_mean=False),
)
add_corr_case(
    "corr-upstream-n4-zm-true",
    "test_corr_swap_correlation",
    [4, "true"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
)
add_corr_case(
    "corr-upstream-n4-zm-false",
    "test_corr_swap_correlation",
    [4, "false"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
)
add_corr_case(
    "corr-upstream-n6-zm-true",
    "test_corr_swap_correlation",
    [6, "true"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=6, assume_zero_mean=True),
)
add_corr_case(
    "corr-upstream-n6-zm-false",
    "test_corr_swap_correlation",
    [6, "false"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=6, assume_zero_mean=False),
)
add_corr_case(
    "corr-upstream-n10-zm-true",
    "test_corr_swap_correlation",
    [10, "true"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=10, assume_zero_mean=True),
)
add_corr_case(
    "corr-upstream-n10-zm-false",
    "test_corr_swap_correlation",
    [10, "false"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES,
    UP_CORR_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=10, assume_zero_mean=False),
)

# Unequal lengths, misaligned calendars, leap day 2020-02-29 (6)
add_corr_case(
    "corr-unequal-len-default-zm-true",
    "synthetic-alignment",
    ["default", "true"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES[:6],
    UP_CORR_Y_VALUES[:6],
    lambda x, y: corr_swap_correlation(x, y, assume_zero_mean=True),
)
add_corr_case(
    "corr-unequal-len-default-zm-false",
    "synthetic-alignment",
    ["default", "false"],
    UP_CORR_DATES,
    UP_CORR_X_VALUES,
    UP_CORR_DATES[:6],
    UP_CORR_Y_VALUES[:6],
    lambda x, y: corr_swap_correlation(x, y, assume_zero_mean=False),
)
add_corr_case(
    "corr-misaligned-n4-zm-true",
    "synthetic-alignment",
    [4, "true"],
    MIS_X_DATES,
    MIS_X_VALUES,
    MIS_Y_DATES,
    MIS_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
)
add_corr_case(
    "corr-misaligned-n4-zm-false",
    "synthetic-alignment",
    [4, "false"],
    MIS_X_DATES,
    MIS_X_VALUES,
    MIS_Y_DATES,
    MIS_Y_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
)
add_corr_case(
    "corr-leap-2020-n4-zm-true",
    "synthetic-calendar",
    [4, "true"],
    LEAP_DATES,
    LEAP_X,
    LEAP_DATES,
    LEAP_Y,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
)
add_corr_case(
    "corr-leap-2020-n4-zm-false",
    "synthetic-calendar",
    [4, "false"],
    LEAP_DATES,
    LEAP_X,
    LEAP_DATES,
    LEAP_Y,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
)

# Empty, singleton, two-point combinations (9)
add_corr_case(
    "corr-empty-empty-default",
    "synthetic-shapes",
    ["default", "true"],
    [],
    [],
    [],
    [],
    lambda x, y: corr_swap_correlation(x, y),
)
add_corr_case(
    "corr-empty-empty-n0",
    "synthetic-shapes",
    [0, "true"],
    [],
    [],
    [],
    [],
    lambda x, y: corr_swap_correlation(x, y, n_days=0),
)
add_corr_case(
    "corr-empty-left-default",
    "synthetic-shapes",
    ["default", "true"],
    [],
    [],
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y),
)
add_corr_case(
    "corr-empty-right-default-err",
    "synthetic-shapes",
    ["default", "true"],
    D6,
    P6,
    [],
    [],
    lambda x, y: corr_swap_correlation(x, y),
    rust_error="Window",
)
add_corr_case(
    "corr-empty-right-n4-zm-true-err",
    "synthetic-shapes",
    [4, "true"],
    D6,
    P6,
    [],
    [],
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
    rust_error="AnnualizationFactor",
)
add_corr_case(
    "corr-empty-right-n4-zm-false",
    "synthetic-shapes",
    [4, "false"],
    D6,
    P6,
    [],
    [],
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
)
add_corr_case(
    "corr-singleton-default-err",
    "synthetic-shapes",
    ["default", "true"],
    SINGLE_DATES,
    SINGLE_VALUES,
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x, y: corr_swap_correlation(x, y),
    rust_error="Window",
)
add_corr_case(
    "corr-two-point-default-zm-true-err",
    "synthetic-shapes",
    ["default", "true"],
    TWO_DATES,
    TWO_X,
    TWO_DATES,
    TWO_Y,
    lambda x, y: corr_swap_correlation(x, y, assume_zero_mean=True),
    rust_error="AnnualizationFactor",
)
add_corr_case(
    "corr-two-point-default-zm-false",
    "synthetic-shapes",
    ["default", "false"],
    TWO_DATES,
    TWO_X,
    TWO_DATES,
    TWO_Y,
    lambda x, y: corr_swap_correlation(x, y, assume_zero_mean=False),
)

# Window & ramp boundaries (9)
add_corr_case(
    "corr-n0-window-err",
    "synthetic-boundaries",
    [0, "true"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=0),
    rust_error="Window",
)
add_corr_case(
    "corr-n1-window-err",
    "synthetic-boundaries",
    [1, "true"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=1),
    rust_error="Window",
)
add_corr_case(
    "corr-n2-zm-true",
    "synthetic-boundaries",
    [2, "true"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=2, assume_zero_mean=True),
)
add_corr_case(
    "corr-n2-zm-false",
    "synthetic-boundaries",
    [2, "false"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=2, assume_zero_mean=False),
)
add_corr_case(
    "corr-oversize-len-plus-1-zm-true-err",
    "synthetic-boundaries",
    [7, "true"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=7, assume_zero_mean=True),
    rust_error="AnnualizationFactor",
)
add_corr_case(
    "corr-oversize-len-plus-1-zm-false",
    "synthetic-boundaries",
    [7, "false"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=7, assume_zero_mean=False),
)
add_corr_case(
    "corr-oversize-len-plus-2-zm-true-err",
    "synthetic-boundaries",
    [8, "true"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=8, assume_zero_mean=True),
    rust_error="Ramp",
)
add_corr_case(
    "corr-oversize-len-plus-2-zm-false",
    "synthetic-boundaries",
    [8, "false"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=8, assume_zero_mean=False),
)
add_corr_case(
    "corr-oversize-len-plus-3-ramp-err",
    "synthetic-boundaries",
    [9, "false"],
    D6,
    P6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=9, assume_zero_mean=False),
    rust_error="Ramp",
)

# Irregular 3-day spacing, NaN, constants, non-positive errors (10)
add_corr_case(
    "corr-irregular-3d-zm-true-err",
    "synthetic-calendar",
    [4, "true"],
    IRREG_DATES,
    IRREG_X,
    IRREG_DATES,
    IRREG_Y,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
    rust_error="AnnualizationFactor",
)
add_corr_case(
    "corr-irregular-3d-zm-false",
    "synthetic-calendar",
    [4, "false"],
    IRREG_DATES,
    IRREG_X,
    IRREG_DATES,
    IRREG_Y,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
)
add_corr_case(
    "corr-nan-mid-n4-zm-true",
    "synthetic-values",
    [4, "true"],
    D8,
    NAN_P8,
    D8,
    NAN_Q8,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
)
add_corr_case(
    "corr-nan-mid-n4-zm-false",
    "synthetic-values",
    [4, "false"],
    D8,
    NAN_P8,
    D8,
    NAN_Q8,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
)
add_corr_case(
    "corr-nan-sparse-n4-zm-true-err",
    "synthetic-values",
    [4, "true"],
    D6,
    NAN_P6,
    D6,
    NAN_Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
    rust_error="AnnualizationFactor",
)
add_corr_case(
    "corr-constant-price-n4-zm-true",
    "synthetic-values",
    [4, "true"],
    D6,
    CONST6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
)
add_corr_case(
    "corr-constant-price-n4-zm-false",
    "synthetic-values",
    [4, "false"],
    D6,
    CONST6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
)
add_corr_case(
    "corr-constant-logret-n4-zm-true",
    "synthetic-values",
    [4, "true"],
    D6,
    CONST_LOGRET6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=True),
)
add_corr_case(
    "corr-constant-logret-n4-zm-false",
    "synthetic-values",
    [4, "false"],
    D6,
    CONST_LOGRET6,
    D6,
    Q6,
    lambda x, y: corr_swap_correlation(x, y, n_days=4, assume_zero_mean=False),
    upstream_note="pandas_rolling_corr_zero_var_inf",
)
add_corr_case(
    "corr-nonpositive-x-err",
    "synthetic-errors",
    [3, "true"],
    NONPOS_ZERO_DATES,
    NONPOS_ZERO_VALUES,
    NONPOS_ZERO_DATES,
    Q6[:4],
    lambda x, y: corr_swap_correlation(x, y, n_days=3, assume_zero_mean=True),
    rust_error="NonPositivePrice",
)
add_corr_case(
    "corr-nonpositive-y-err",
    "synthetic-errors",
    [3, "false"],
    NONPOS_ZERO_DATES,
    P6[:4],
    NONPOS_ZERO_DATES,
    NONPOS_NEG_VALUES,
    lambda x, y: corr_swap_correlation(x, y, n_days=3, assume_zero_mean=False),
    rust_error="NonPositivePrice",
)

payload = {
    "schema": "gsquant-volcorr-swap-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/econometrics.py:vol_swap_volatility",
        "gs_quant/timeseries/econometrics.py:corr_swap_correlation",
        "gs_quant/timeseries/econometrics.py:volatility",
        "gs_quant/timeseries/econometrics.py:correlation",
        "gs_quant/timeseries/econometrics.py:_get_annualization_factor",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "volcorr-swap-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream volcorr_swap cases")
