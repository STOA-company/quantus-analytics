"""Generate upstream correlation parity vectors only. Run with the locked upstream Python environment."""
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys

# The caller supplies the fixed checkout; no network or Rust execution here.
sys.path.insert(0, sys.argv[1])
import pandas as pd
from gs_quant.timeseries import Returns, SeriesType, Window, correlation

ROOT = pathlib.Path(__file__).resolve().parent

PRICES_DATES = [dt.date(2019, 1, i) for i in range(1, 7)]
PRICES_VALUES = [100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792]

IND_X_DATES = [
    dt.date(2024, 2, 26),
    dt.date(2024, 2, 28),
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 1),
    dt.date(2024, 3, 4),
    dt.date(2024, 3, 6),
]
IND_X_VALUES = [0.01, 0.03, float("nan"), 0.02, 0.01, 0.04]

IND_Y_DATES = [
    dt.date(2024, 2, 26),
    dt.date(2024, 2, 27),
    dt.date(2024, 2, 28),
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 4),
    dt.date(2024, 3, 6),
]
IND_Y_VALUES = [0.02, 0.03, -0.02, 0.04, 0.03, 0.01]

IND_X_DENSE_VALUES = [0.01, 0.03, 0.02, 0.02, 0.01, 0.04]
IND_Y_DENSE_VALUES = [0.02, -0.02, 0.04, 0.01, 0.03, 0.01]
IND_ZEROS_VALUES = [0.0] * 6

BND_X_DATES = [dt.date(2024, 3, d) for d in (1, 2, 3, 5, 6)]
BND_X_VALUES = [10.0, 12.0, 14.0, 13.0, 16.0]
BND_Y_DATES = [dt.date(2024, 3, d) for d in (1, 3, 4, 5, 6)]
BND_Y_VALUES = [20.0, 24.0, 22.0, 19.0, 25.0]
BND_GAP_VALUES = [0.01, float("nan"), 0.02, 0.04, 0.01]
BND_CONST_VALUES = [0.01] * 5


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
    y_dates,
    y_values,
    fn,
    rust_error=None,
    upstream_note=None,
    empty_index_kind=None,
):
    sx = (
        pd.Series(x_values, index=x_dates, dtype=float)
        if x_dates
        else (
            pd.Series([], index=pd.Index([], dtype=object), dtype=float)
            if empty_index_kind == "Index"
            else pd.Series(dtype=float)
        )
    )
    sy = pd.Series(y_values, index=y_dates, dtype=float) if y_dates else pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "correlation",
        "args": args,
        "input_x": pack_series(x_dates, x_values, sx),
        "input_y": pack_series(y_dates, y_values, sy),
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note
    try:
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


# 5 indicator outputs + 1 indicator error from upstream-indicators.json
add_case(
    "correlation-returns-2-observations",
    "upstream-indicators.json",
    ["observations", 2, 0, "none", "none", "false"],
    IND_X_DATES,
    IND_X_VALUES,
    IND_Y_DATES,
    IND_Y_VALUES,
    lambda x, y: correlation(x, y, Window(2, 0), type_=SeriesType.RETURNS),
)
add_case(
    "correlation-returns-zero-mean",
    "upstream-indicators.json",
    ["observations", 2, 0, "none", "none", "true"],
    IND_X_DATES,
    IND_X_DENSE_VALUES,
    IND_X_DATES,
    IND_Y_DENSE_VALUES,
    lambda x, y: correlation(x, y, Window(2, 0), type_=SeriesType.RETURNS, assume_zero_mean=True),
)
add_case(
    "correlation-returns-3d",
    "upstream-indicators.json",
    ["days", 3, 0, "none", "none", "false"],
    IND_X_DATES,
    IND_X_VALUES,
    IND_Y_DATES,
    IND_Y_VALUES,
    lambda x, y: correlation(x, y, Window("3d", 0), type_=SeriesType.RETURNS),
)
add_case(
    "correlation-returns-zero-denominator",
    "upstream-indicators.json",
    ["observations", 2, 0, "none", "none", "false"],
    IND_X_DATES,
    IND_X_DENSE_VALUES,
    IND_X_DATES,
    IND_ZEROS_VALUES,
    lambda x, y: correlation(x, y, Window(2, 0), type_=SeriesType.RETURNS),
)
add_case(
    "correlation-prices-default",
    "upstream-indicators.json",
    ["expanding", 0, 0, "simple", "simple", "false"],
    PRICES_DATES,
    PRICES_VALUES,
    PRICES_DATES,
    PRICES_VALUES,
    lambda x, y: correlation(x, y),
)
add_case(
    "correlation-returns-zero-mean-irregular",
    "upstream-indicators.json",
    ["observations", 2, 0, "none", "none", "true"],
    IND_X_DATES,
    IND_X_VALUES,
    IND_Y_DATES,
    IND_Y_VALUES,
    lambda x, y: correlation(x, y, Window(2, 0), type_=SeriesType.RETURNS, assume_zero_mean=True),
    rust_error="AnnualizationFactor",
)

# 9 synthetic boundary cases from boundaries.py / boundaries.json
add_case(
    "prices-mixed",
    "boundaries.json",
    ["observations", 3, 0, "logarithmic", "absolute", "false"],
    BND_X_DATES,
    BND_X_VALUES,
    BND_Y_DATES,
    BND_Y_VALUES,
    lambda x, y: correlation(
        x, y, Window(3, 0), returns_type=(Returns.LOGARITHMIC, Returns.ABSOLUTE)
    ),
)
add_case(
    "prices-observation-ramp",
    "boundaries.json",
    ["observations", 3, 2, "logarithmic", "absolute", "false"],
    BND_X_DATES,
    BND_X_VALUES,
    BND_Y_DATES,
    BND_Y_VALUES,
    lambda x, y: correlation(
        x, y, Window(3, 2), returns_type=(Returns.LOGARITHMIC, Returns.ABSOLUTE)
    ),
)
add_case(
    "prices-calendar-ramp",
    "boundaries.json",
    ["days", 3, 2, "logarithmic", "absolute", "false"],
    BND_X_DATES,
    BND_X_VALUES,
    BND_Y_DATES,
    BND_Y_VALUES,
    lambda x, y: correlation(
        x, y, Window("3d", "2d"), returns_type=(Returns.LOGARITHMIC, Returns.ABSOLUTE)
    ),
)
add_case(
    "returns-gap",
    "boundaries.json",
    ["observations", 3, 0, "none", "none", "false"],
    BND_X_DATES,
    BND_GAP_VALUES,
    BND_Y_DATES,
    BND_Y_VALUES,
    lambda x, y: correlation(x, y, Window(3, 0), type_=SeriesType.RETURNS),
)
add_case(
    "returns-constant",
    "boundaries.json",
    ["observations", 3, 0, "none", "none", "false"],
    BND_X_DATES,
    BND_GAP_VALUES,
    BND_X_DATES,
    BND_CONST_VALUES,
    lambda x, y: correlation(x, y, Window(3, 0), type_=SeriesType.RETURNS),
    upstream_note="pandas_rolling_corr_zero_var_inf",
)
add_case(
    "returns-no-common",
    "boundaries.json",
    ["observations", 3, 0, "none", "none", "false"],
    BND_X_DATES,
    BND_X_VALUES,
    BND_Y_DATES[2:3],
    BND_Y_VALUES[2:3],
    lambda x, y: correlation(x, y, Window(3, 0), type_=SeriesType.RETURNS),
)
add_case(
    "returns-zero-mean-no-common",
    "boundaries.json",
    ["observations", 3, 0, "none", "none", "true"],
    BND_X_DATES,
    BND_X_VALUES,
    BND_Y_DATES[2:3],
    BND_Y_VALUES[2:3],
    lambda x, y: correlation(
        x, y, Window(3, 0), type_=SeriesType.RETURNS, assume_zero_mean=True
    ),
    rust_error="AnnualizationFactor",
)
add_case(
    "returns-zero-mean-empty-left",
    "boundaries.json",
    ["observations", 3, 0, "none", "none", "true"],
    [],
    [],
    BND_Y_DATES,
    BND_Y_VALUES,
    lambda x, y: correlation(
        x, y, Window(3, 0), type_=SeriesType.RETURNS, assume_zero_mean=True
    ),
    empty_index_kind="Index",
)
add_case(
    "returns-zero-mean-one-common",
    "boundaries.json",
    ["observations", 3, 0, "none", "none", "true"],
    BND_X_DATES,
    BND_X_VALUES,
    BND_Y_DATES[0:1],
    BND_Y_VALUES[0:1],
    lambda x, y: correlation(
        x, y, Window(3, 0), type_=SeriesType.RETURNS, assume_zero_mean=True
    ),
    rust_error="AnnualizationFactor",
)

payload = {
    "schema": "gsquant-correlation-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/econometrics.py:correlation",
        "gs_quant/timeseries/econometrics.py:volatility",
        "gs_quant/timeseries/econometrics.py:_get_annualization_factor",
        "gs_quant/timeseries/statistics.py:mean",
        "gs_quant/timeseries/datetime.py:interpolate",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "correlation-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream correlation cases")
