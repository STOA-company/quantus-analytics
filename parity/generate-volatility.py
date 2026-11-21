"""Generate upstream volatility parity vectors only. Run with the locked upstream Python environment."""
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys

# The caller supplies the fixed checkout; no network or Rust execution here.
sys.path.insert(0, sys.argv[1])
import pandas as pd
from gs_quant.timeseries import Returns, Window, volatility

ROOT = pathlib.Path(__file__).resolve().parent

PRICES_DATES = [dt.date(2019, 1, i) for i in range(1, 7)]
PRICES_VALUES = [100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792]

X_DATES = [
    dt.date(2024, 2, 26),
    dt.date(2024, 2, 28),
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 1),
    dt.date(2024, 3, 4),
    dt.date(2024, 3, 6),
]
X_VALUES = [0.01, 0.03, float("nan"), 0.02, 0.01, 0.04]

BASE4_DATES = [
    dt.date(2024, 2, 26),
    dt.date(2024, 2, 27),
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 1),
]
BASE4_VALUES = [0.01, 0.03, float("nan"), 0.05]

SINGLE_DATES = [dt.date(2024, 2, 26)]
SINGLE_VALUES = [0.01]

STEP_START = dt.date(2024, 2, 26)
STEP_VALUES = [0.01, 0.03, 0.02]


def pack(v: float):
    if math.isnan(v):
        return {"special": "NaN"}
    if math.isinf(v):
        return {"special": "+Inf" if v > 0 else "-Inf"}
    return float(v)


cases = []


def add_case(case_id, source_ref, args, dates, values, fn, rust_error=None):
    series = pd.Series(values, index=dates, dtype=float) if dates else pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "volatility",
        "args": args,
        "input": {
            "index_kind": "Index[date]" if dates else type(series.index).__name__,
            "timezone": None,
            "dates": [str(d) for d in dates],
            "values": [pack(v) for v in values],
            "dtype": "float64",
        },
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


# 6 indicator cases from upstream-indicators.json (3m calendar-month case excluded)
add_case(
    "volatility-prices-default",
    "upstream-indicators.json",
    ["expanding", 0, 0, "simple", "auto", "false"],
    PRICES_DATES,
    PRICES_VALUES,
    lambda x: volatility(x),
)
add_case(
    "volatility-prices-3d-ramp",
    "upstream-indicators.json",
    ["days", 3, 3, "simple", "auto", "false"],
    PRICES_DATES,
    PRICES_VALUES,
    lambda x: volatility(x, "3d"),
)
add_case(
    "volatility-log-prices-3-observations",
    "upstream-indicators.json",
    ["observations", 3, 0, "logarithmic", "auto", "true"],
    PRICES_DATES,
    PRICES_VALUES,
    lambda x: volatility(x, Window(3, 0), returns_type=Returns.LOGARITHMIC, assume_zero_mean=True),
)
add_case(
    "volatility-returns-3-observations",
    "upstream-indicators.json",
    ["observations", 3, 0, "none", "auto", "false"],
    X_DATES,
    X_VALUES,
    lambda x: volatility(x, Window(3, 0), returns_type=None),
)
add_case(
    "volatility-returns-zero-mean",
    "upstream-indicators.json",
    ["observations", 3, 0, "none", "auto", "true"],
    X_DATES,
    X_VALUES,
    lambda x: volatility(x, Window(3, 0), returns_type=None, assume_zero_mean=True),
)
add_case(
    "volatility-returns-3d",
    "upstream-indicators.json",
    ["days", 3, 0, "none", "auto", "false"],
    X_DATES,
    X_VALUES,
    lambda x: volatility(x, Window("3d", 0), returns_type=None),
)

# 13 synthetic boundary & explicit annualization cases from owner-upstream.json
add_case(
    "empty-default",
    "owner-upstream.json",
    ["expanding", 0, 0, "simple", "auto", "false"],
    [],
    [],
    lambda x: volatility(x),
)
add_case(
    "empty-zero-window",
    "owner-upstream.json",
    ["observations", 0, 0, "none", "auto", "false"],
    [],
    [],
    lambda x: volatility(x, Window(0, 0), returns_type=None),
)
add_case(
    "singleton-auto",
    "owner-upstream.json",
    ["expanding", 0, 0, "none", "auto", "false"],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: volatility(x, returns_type=None),
    rust_error="AnnualizationFactor",
)
add_case(
    "singleton-explicit-252-sample",
    "owner-upstream.json",
    ["expanding", 0, 0, "none", 252, "false"],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: volatility(x, returns_type=None, annualization_factor=252),
)
add_case(
    "singleton-explicit-252-zero-mean",
    "owner-upstream.json",
    ["expanding", 0, 0, "none", 252, "true"],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: volatility(x, returns_type=None, annualization_factor=252, assume_zero_mean=True),
)
add_case(
    "oversize-explicit",
    "owner-upstream.json",
    ["observations", 5, 0, "none", 252, "false"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(x, Window(5, 0), returns_type=None, annualization_factor=252),
)
add_case(
    "oversize-auto",
    "owner-upstream.json",
    ["observations", 5, 0, "none", "auto", "false"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(x, Window(5, 0), returns_type=None),
    rust_error="AnnualizationFactor",
)
add_case(
    "ramp-equals-length",
    "owner-upstream.json",
    ["observations", 4, 4, "none", "auto", "false"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(x, Window(4, 4), returns_type=None),
)
add_case(
    "ramp-exceeds-length",
    "owner-upstream.json",
    ["observations", 4, 5, "none", "auto", "false"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(x, Window(4, 5), returns_type=None),
    rust_error="Ramp",
)
add_case(
    "zero-factor",
    "owner-upstream.json",
    ["days", 3, 0, "none", 0, "false"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(x, Window("3d", 0), returns_type=None, annualization_factor=0),
)
add_case(
    "explicit-annualization-255",
    "owner-upstream.json",
    ["expanding", 0, 0, "none", 255, "false"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(x, returns_type=None, annualization_factor=255),
)
add_case(
    "day-boundary-sample",
    "owner-upstream.json",
    ["days", 3, 1, "none", 252, "false"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(x, Window("3d", "1d"), returns_type=None, annualization_factor=252),
)
add_case(
    "day-boundary-rms",
    "owner-upstream.json",
    ["days", 3, 1, "none", 252, "true"],
    BASE4_DATES,
    BASE4_VALUES,
    lambda x: volatility(
        x, Window("3d", "1d"), returns_type=None, annualization_factor=252, assume_zero_mean=True
    ),
)

# 13 auto-annualization mode and gap cases from owner-upstream.json
AUTO_STEPS = [
    (1, None),
    (2, None),
    (3, "AnnualizationFactor"),
    (6, None),
    (8, "AnnualizationFactor"),
    (14, None),
    (17, "AnnualizationFactor"),
    (25, None),
    (35, "AnnualizationFactor"),
    (85, None),
    (97, "AnnualizationFactor"),
    (360, None),
    (386, "AnnualizationFactor"),
]
for step, err in AUTO_STEPS:
    step_dates = [STEP_START + dt.timedelta(days=k * step) for k in range(3)]
    add_case(
        f"auto-average-{step}-days",
        "owner-upstream.json",
        ["expanding", 0, 0, "none", "auto", "false"],
        step_dates,
        STEP_VALUES,
        lambda x: volatility(x, returns_type=None),
        rust_error=err,
    )

payload = {
    "schema": "gsquant-volatility-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/econometrics.py:volatility",
        "gs_quant/timeseries/econometrics.py:_get_annualization_factor",
        "gs_quant/timeseries/statistics.py:std",
        "gs_quant/timeseries/statistics.py:mean",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "volatility-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream volatility cases")
