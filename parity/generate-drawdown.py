"""Generate upstream max_drawdown parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries import Window, max_drawdown

ROOT = pathlib.Path(__file__).resolve().parent

DAILY_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 3),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 7),
    dt.date(2019, 1, 8),
]
ORIGINAL_VALUES = [1.0, 5.0, 5.0, 4.0, 4.0, 1.0]

GAP5_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 7),
    dt.date(2019, 1, 8),
]
ZERO_VALUES = [0.0, 0.0, 2.0, 0.0, 1.0]
NEGATIVE_VALUES = [-5.0, -2.0, -9.0, -1.0, -5.0]
MISSING_VALUES = [5.0, float("nan"), 2.0, float("nan"), 1.0]

EXT_DATES = [dt.date(2019, 1, d) for d in (1, 2, 3, 4)]
NEG_AFTER_ZERO_VALUES = [0.0, -1.0, -2.0, 0.0]
EXPLICIT_INF_VALUES = [2.0, float("inf"), 1.0, -1.0]
ALL_MISSING_VALUES = [float("nan")] * 4

ONE_DATE = [dt.date(2019, 1, 1)]

SIGNED_ZERO_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 1),
]


def pack(v: float):
    if math.isnan(v):
        return {"special": "NaN"}
    if math.isinf(v):
        return {"special": "+Inf" if v > 0 else "-Inf"}
    return float(v)


def fmt_date(d) -> str:
    if isinstance(d, pd.Timestamp):
        return d.date().isoformat()
    return str(d)


cases = []


def add_case(case_id, source_ref, args, dates, values, fn, rust_error=None):
    index = pd.to_datetime(dates) if dates else pd.DatetimeIndex([])
    series = pd.Series(values, index=index, dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "max_drawdown",
        "args": args,
        "input": {
            "index_kind": type(series.index).__name__,
            "timezone": None,
            "dates": [d.isoformat() for d in dates],
            "values": [pack(v) for v in values],
            "dtype": "float64",
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
                raise AssertionError(
                    f"{case_id}: expected upstream error for {rust_error}, got output"
                )
            out_dtype = str(out.dtype)
            if args[0] == "days" and not dates and len(out) == 0 and out_dtype == "object":
                entry["rust_representation"] = "empty_f64_series"
            entry["output"] = {
                "index_kind": type(out.index).__name__,
                "timezone": None,
                "dates": [fmt_date(d) for d in out.index],
                "values": [pack(float(v)) for v in out],
                "unit": "ratio",
                "dtype": out_dtype,
            }
    cases.append(entry)


# 3 representative cases from upstream test_econometrics.py::test_max_drawdown
add_case(
    "test-max-drawdown-default",
    "test_econometrics.py::test_max_drawdown",
    ["expanding", 0, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x),
)
add_case(
    "test-max-drawdown-obs-2",
    "test_econometrics.py::test_max_drawdown",
    ["observations", 2, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window(2, 0)),
)
add_case(
    "test-max-drawdown-2d",
    "test_econometrics.py::test_max_drawdown",
    ["days", 2, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window("2d", 0)),
)

# 16 bundled synthetic boundary cases from boundaries-1.json
add_case(
    "original-default",
    "boundaries-1.json",
    ["expanding", 0, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x),
)
add_case(
    "original-observations",
    "boundaries-1.json",
    ["observations", 2, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window(2, 0)),
)
add_case(
    "original-days",
    "boundaries-1.json",
    ["days", 2, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window("2d", 0)),
)
add_case(
    "calendar-default-ramp",
    "boundaries-1.json",
    ["days", 2, 2],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, "2d"),
)
add_case(
    "observation-ramp",
    "boundaries-1.json",
    ["observations", 2, 2],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window(2, 2)),
)
add_case(
    "calendar-gaps",
    "boundaries-1.json",
    ["days", 2, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window("2d", 0)),
)
add_case(
    "zero-observations",
    "boundaries-1.json",
    ["observations", 3, 0],
    GAP5_DATES,
    ZERO_VALUES,
    lambda x: max_drawdown(x, Window(3, 0)),
)
add_case(
    "zero-calendar",
    "boundaries-1.json",
    ["days", 3, 0],
    GAP5_DATES,
    ZERO_VALUES,
    lambda x: max_drawdown(x, Window("3d", 0)),
)
add_case(
    "negative-observations",
    "boundaries-1.json",
    ["observations", 3, 0],
    GAP5_DATES,
    NEGATIVE_VALUES,
    lambda x: max_drawdown(x, Window(3, 0)),
)
add_case(
    "negative-calendar",
    "boundaries-1.json",
    ["days", 3, 0],
    GAP5_DATES,
    NEGATIVE_VALUES,
    lambda x: max_drawdown(x, Window("3d", 0)),
)
add_case(
    "missing-observations",
    "boundaries-1.json",
    ["observations", 3, 0],
    GAP5_DATES,
    MISSING_VALUES,
    lambda x: max_drawdown(x, Window(3, 0)),
)
add_case(
    "missing-calendar",
    "boundaries-1.json",
    ["days", 3, 0],
    GAP5_DATES,
    MISSING_VALUES,
    lambda x: max_drawdown(x, Window("3d", 0)),
)
add_case(
    "oversize-observations",
    "boundaries-1.json",
    ["observations", 10, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window(10, 0)),
)
add_case(
    "oversize-ramp",
    "boundaries-1.json",
    ["observations", 10, 10],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window(10, 10)),
    rust_error="Ramp",
)
add_case(
    "empty-default",
    "boundaries-1.json",
    ["expanding", 0, 0],
    [],
    [],
    lambda x: max_drawdown(x),
)
add_case(
    "empty-calendar",
    "boundaries-1.json",
    ["days", 2, 0],
    [],
    [],
    lambda x: max_drawdown(x, Window("2d", 0)),
)

# 16 extended boundary cases from extended-boundaries.txt
for prefix, vals in [
    ("neg-after-zero", NEG_AFTER_ZERO_VALUES),
    ("explicit-inf", EXPLICIT_INF_VALUES),
    ("all-missing", ALL_MISSING_VALUES),
]:
    add_case(
        f"{prefix}-default",
        "extended-boundaries.txt",
        ["expanding", 0, 0],
        EXT_DATES,
        vals,
        lambda x: max_drawdown(x),
    )
    add_case(
        f"{prefix}-observations",
        "extended-boundaries.txt",
        ["observations", 2, 0],
        EXT_DATES,
        vals,
        lambda x: max_drawdown(x, Window(2, 0)),
    )
    add_case(
        f"{prefix}-days",
        "extended-boundaries.txt",
        ["days", 2, 0],
        EXT_DATES,
        vals,
        lambda x: max_drawdown(x, Window("2d", 0)),
    )
    add_case(
        f"{prefix}-days-ramp",
        "extended-boundaries.txt",
        ["days", 2, 2],
        EXT_DATES,
        vals,
        lambda x: max_drawdown(x, "2d"),
    )

add_case(
    "empty-extended-default",
    "extended-boundaries.txt",
    ["expanding", 0, 0],
    [],
    [],
    lambda x: max_drawdown(x),
)
add_case(
    "empty-extended-observations",
    "extended-boundaries.txt",
    ["observations", 2, 0],
    [],
    [],
    lambda x: max_drawdown(x, Window(2, 0)),
)
add_case(
    "empty-extended-days",
    "extended-boundaries.txt",
    ["days", 2, 0],
    [],
    [],
    lambda x: max_drawdown(x, Window("2d", 0)),
)
add_case(
    "empty-extended-days-ramp",
    "extended-boundaries.txt",
    ["days", 2, 2],
    [],
    [],
    lambda x: max_drawdown(x, "2d"),
    rust_error="Ramp",
)

# 6 additional boundary probes from one-point-boundaries.txt and zero-window validation
add_case(
    "one-point-positive-default",
    "one-point-boundaries.txt",
    ["expanding", 0, 0],
    ONE_DATE,
    [5.0],
    lambda x: max_drawdown(x),
)
add_case(
    "one-point-positive-oversize",
    "one-point-boundaries.txt",
    ["observations", 2, 0],
    ONE_DATE,
    [5.0],
    lambda x: max_drawdown(x, Window(2, 0)),
)
add_case(
    "one-point-positive-days",
    "one-point-boundaries.txt",
    ["days", 2, 0],
    ONE_DATE,
    [5.0],
    lambda x: max_drawdown(x, Window("2d", 0)),
)
add_case(
    "one-point-zero-default",
    "one-point-boundaries.txt",
    ["expanding", 0, 0],
    ONE_DATE,
    [0.0],
    lambda x: max_drawdown(x),
)
add_case(
    "one-point-negative-default",
    "one-point-boundaries.txt",
    ["expanding", 0, 0],
    ONE_DATE,
    [-5.0],
    lambda x: max_drawdown(x),
)
add_case(
    "invalid-window-zero",
    "boundaries-extra",
    ["observations", 0, 0],
    DAILY_DATES,
    ORIGINAL_VALUES,
    lambda x: max_drawdown(x, Window(0, 0)),
    rust_error="Window",
)

# Signed-zero tie-breaking boundaries (both zero orderings, all-negative zero, NaN-interleaved)
for prefix, vals in [
    ("signed-zero-pos-neg", [0.0, -0.0, -1.0, -2.0]),
    ("signed-zero-neg-pos", [-0.0, 0.0, -1.0, -2.0]),
    ("signed-zero-neg-neg", [-0.0, -0.0, -1.0, -2.0]),
]:
    add_case(
        f"{prefix}-observations-2",
        "signed-zero-python.txt",
        ["observations", 2, 0],
        SIGNED_ZERO_DATES,
        vals,
        lambda x: max_drawdown(x, Window(2, 0)),
    )
    add_case(
        f"{prefix}-days-2",
        "signed-zero-python.txt",
        ["days", 2, 0],
        SIGNED_ZERO_DATES,
        vals,
        lambda x: max_drawdown(x, Window("2d", 0)),
    )
    add_case(
        f"{prefix}-days-3",
        "signed-zero-python.txt",
        ["days", 3, 0],
        SIGNED_ZERO_DATES,
        vals,
        lambda x: max_drawdown(x, Window("3d", 0)),
    )

add_case(
    "signed-zero-pos-neg-nan-days-4",
    "signed-zero-peak-python.txt",
    ["days", 4, 0],
    SIGNED_ZERO_DATES,
    [0.0, -0.0, float("nan"), -1.0],
    lambda x: max_drawdown(x, Window("4d", 0)),
)
add_case(
    "signed-zero-neg-pos-nan-days-4",
    "signed-zero-peak-python.txt",
    ["days", 4, 0],
    SIGNED_ZERO_DATES,
    [-0.0, 0.0, float("nan"), -1.0],
    lambda x: max_drawdown(x, Window("4d", 0)),
)

payload = {
    "schema": "gsquant-drawdown-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/econometrics.py:max_drawdown",
        "gs_quant/timeseries/helper.py:normalize_window",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "drawdown-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream drawdown cases")
