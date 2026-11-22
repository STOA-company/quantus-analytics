"""Generate upstream relative_strength_index parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries import Window, relative_strength_index

ROOT = pathlib.Path(__file__).resolve().parent

SPX_DATES = [
    dt.date(2020, 1, 2),
    dt.date(2020, 1, 3),
    dt.date(2020, 1, 6),
    dt.date(2020, 1, 7),
    dt.date(2020, 1, 8),
    dt.date(2020, 1, 9),
    dt.date(2020, 1, 10),
    dt.date(2020, 1, 13),
    dt.date(2020, 1, 14),
    dt.date(2020, 1, 15),
    dt.date(2020, 1, 16),
    dt.date(2020, 1, 17),
    dt.date(2020, 1, 21),
    dt.date(2020, 1, 22),
    dt.date(2020, 1, 23),
    dt.date(2020, 1, 24),
    dt.date(2020, 1, 27),
    dt.date(2020, 1, 28),
    dt.date(2020, 1, 29),
    dt.date(2020, 1, 30),
    dt.date(2020, 1, 31),
    dt.date(2020, 2, 3),
]

SPX_VALUES = [
    3257.8501,
    3234.8501,
    3246.28,
    3237.1799,
    3253.05,
    3274.7,
    3265.3501,
    3288.1299,
    3283.1499,
    3289.29,
    3316.8101,
    3329.6201,
    3320.79,
    3321.75,
    3325.54,
    3295.47,
    3243.6299,
    3276.24,
    3273.3999,
    3283.6599,
    3225.52,
    3248.9199,
]

JAN10_DATES = [dt.date(2020, 1, day) for day in range(1, 11)]
JAN6_DATES = JAN10_DATES[:6]

MIXED_VALUES = [10.0, 12.0, 11.0, 13.0, 13.0, 9.0, 10.5, 14.0, 12.0, 15.0]

# Irregular dates crossing leap day 2020-02-29
IRREGULAR_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 4),
    dt.date(2020, 3, 7),
    dt.date(2020, 3, 10),
    dt.date(2020, 3, 14),
    dt.date(2020, 3, 20),
]
IRREGULAR_VALUES = [10.0, 12.0, 11.0, 14.0, 12.0, 12.0, 15.0, 9.0]


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


cases = []


def add_case(case_id, source_ref, args, dates, values, fn, rust_error=None):
    # Upstream relative_strength_index delegates to smoothed_moving_average, whose calendar-day
    # branch compares (x.index[i] - window_size).date() and therefore requires a datetime.date
    # index (pandas DatetimeIndex raises TypeError).
    if dates:
        series = pd.Series(values, index=dates, dtype=float)
    else:
        series = pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "relative_strength_index",
        "args": args,
        "input": {
            "index_kind": index_kind(series.index),
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
            entry["output"] = {
                "index_kind": index_kind(out.index),
                "timezone": None,
                "dates": [fmt_date(d) for d in out.index],
                "values": [pack(float(v)) for v in out],
                "unit": "rsi",
                "dtype": str(out.dtype),
            }
    cases.append(entry)


# 3 cases from upstream test_technicals.py::{test_relative_strength_index, test_relative_strength_index_date_offset_window}
add_case(
    "test-rsi-spx-default",
    "test_technicals.py::test_relative_strength_index",
    ["default", 0, 0],
    SPX_DATES,
    SPX_VALUES,
    lambda x: relative_strength_index(x),
)
add_case(
    "test-rsi-increasing-14",
    "test_technicals.py::test_relative_strength_index",
    ["observations", 14, 14],
    SPX_DATES,
    [float(v) for v in range(1, 23)],
    lambda x: relative_strength_index(x, 14),
)
add_case(
    "test-rsi-3d-increasing",
    "test_technicals.py::test_relative_strength_index_date_offset_window",
    ["days", 3, 3],
    JAN10_DATES,
    [float(v) for v in range(1, 11)],
    lambda x: relative_strength_index(x, "3d"),
)

# Synthetic boundary cases mirroring pinned-boundaries-1.txt
add_case(
    "inc-obs3",
    "pinned-boundaries-1.txt",
    ["observations", 3, 3],
    JAN10_DATES,
    [float(v) for v in range(1, 11)],
    lambda x: relative_strength_index(x, 3),
)
add_case(
    "dec-obs3",
    "pinned-boundaries-1.txt",
    ["observations", 3, 3],
    JAN10_DATES,
    [float(v) for v in range(10, 0, -1)],
    lambda x: relative_strength_index(x, 3),
)
add_case(
    "const-obs3",
    "pinned-boundaries-1.txt",
    ["observations", 3, 3],
    JAN10_DATES,
    [5.0] * 10,
    lambda x: relative_strength_index(x, 3),
)
add_case(
    "mixed-obs3-r0",
    "pinned-boundaries-1.txt",
    ["observations", 3, 0],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window(3, 0)),
)
add_case(
    "mixed-obs3-r3",
    "pinned-boundaries-1.txt",
    ["observations", 3, 3],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window(3, 3)),
)
add_case(
    "mixed-default-14-short",
    "pinned-boundaries-1.txt",
    ["default", 0, 0],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x),
    rust_error="Ramp",
)
add_case(
    "mixed-obs1",
    "pinned-boundaries-1.txt",
    ["observations", 1, 0],
    JAN10_DATES[:5],
    MIXED_VALUES[:5],
    lambda x: relative_strength_index(x, Window(1, 0)),
)
add_case(
    "irreg-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 2],
    IRREGULAR_DATES,
    IRREGULAR_VALUES,
    lambda x: relative_strength_index(x, 2),
)
add_case(
    "irreg-4d",
    "pinned-boundaries-1.txt",
    ["days", 4, 4],
    IRREGULAR_DATES,
    IRREGULAR_VALUES,
    lambda x: relative_strength_index(x, "4d"),
)
add_case(
    "irreg-4d-r0",
    "pinned-boundaries-1.txt",
    ["days", 4, 0],
    IRREGULAR_DATES,
    IRREGULAR_VALUES,
    lambda x: relative_strength_index(x, Window("4d", 0)),
)
add_case(
    "nan-mid-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    JAN6_DATES,
    [1.0, 2.0, float("nan"), 4.0, 3.0, 5.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "inf-mid-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    JAN6_DATES,
    [1.0, 2.0, float("inf"), 4.0, 3.0, 5.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "empty-obs3",
    "pinned-boundaries-1.txt",
    ["observations", 3, 3],
    [],
    [],
    lambda x: relative_strength_index(x, 3),
)
add_case(
    "empty-3d",
    "pinned-boundaries-1.txt",
    ["days", 3, 3],
    [],
    [],
    lambda x: relative_strength_index(x, "3d"),
    rust_error="Ramp",
)
add_case(
    "single-obs1",
    "pinned-boundaries-1.txt",
    ["observations", 1, 0],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x, Window(1, 0)),
)
add_case(
    "two-obs1",
    "pinned-boundaries-1.txt",
    ["observations", 1, 0],
    JAN10_DATES[:2],
    [5.0, 6.0],
    lambda x: relative_strength_index(x, Window(1, 0)),
)
add_case(
    "two-obs1-r1",
    "pinned-boundaries-1.txt",
    ["observations", 1, 1],
    JAN10_DATES[:2],
    [5.0, 6.0],
    lambda x: relative_strength_index(x, Window(1, 1)),
)
add_case(
    "oversize-ramp",
    "pinned-boundaries-1.txt",
    ["observations", 3, 10],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window(3, 10)),
    rust_error="Ramp",
)
add_case(
    "zero-window",
    "pinned-boundaries-1.txt",
    ["observations", 0, 0],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window(0, 0)),
    rust_error="Window",
)

# Additional boundary probes (Window::Expanding, calendar windows, ±Inf/NaN, len 0/1/2, ramp == len(x))
add_case(
    "mixed-expanding",
    "boundaries-extra",
    ["expanding", 0, 0],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window(None, 0)),
)
add_case(
    "empty-expanding",
    "boundaries-extra",
    ["expanding", 0, 0],
    [],
    [],
    lambda x: relative_strength_index(x, Window(None, 0)),
)
add_case(
    "single-expanding",
    "boundaries-extra",
    ["expanding", 0, 0],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x, Window(None, 0)),
)
add_case(
    "two-expanding",
    "boundaries-extra",
    ["expanding", 0, 0],
    JAN10_DATES[:2],
    [5.0, 6.0],
    lambda x: relative_strength_index(x, Window(None, 0)),
)
add_case(
    "spx-obs14-r0",
    "boundaries-extra",
    ["observations", 14, 0],
    SPX_DATES,
    SPX_VALUES,
    lambda x: relative_strength_index(x, Window(14, 0)),
)
add_case(
    "mixed-3d-r3",
    "boundaries-extra",
    ["days", 3, 3],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, "3d"),
)
add_case(
    "mixed-3d-r0",
    "boundaries-extra",
    ["days", 3, 0],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window("3d", 0)),
)
add_case(
    "irreg-3d-r3",
    "boundaries-extra",
    ["days", 3, 3],
    IRREGULAR_DATES,
    IRREGULAR_VALUES,
    lambda x: relative_strength_index(x, "3d"),
)
add_case(
    "irreg-4d-r2d",
    "boundaries-extra",
    ["days", 4, 2],
    IRREGULAR_DATES,
    IRREGULAR_VALUES,
    lambda x: relative_strength_index(x, Window("4d", "2d")),
)
add_case(
    "nan-lead-obs2",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN6_DATES[:5],
    [float("nan"), 2.0, 3.0, 1.0, 4.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "nan-all-obs2",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN6_DATES,
    [float("nan")] * 6,
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "ninf-mid-obs2",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN6_DATES,
    [1.0, 2.0, float("-inf"), 4.0, 3.0, 5.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "inf-lead-obs2",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN6_DATES[:4],
    [1.0, float("inf"), 2.0, 3.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "ninf-lead-obs2",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN6_DATES[:4],
    [1.0, float("-inf"), 2.0, 3.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "inf-inf-obs2",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN6_DATES[:4],
    [1.0, float("inf"), float("inf"), 3.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "empty-default",
    "boundaries-extra",
    ["default", 0, 0],
    [],
    [],
    lambda x: relative_strength_index(x),
)
add_case(
    "empty-3d-r0",
    "boundaries-extra",
    ["days", 3, 0],
    [],
    [],
    lambda x: relative_strength_index(x, Window("3d", 0)),
)
add_case(
    "empty-obs0",
    "boundaries-extra",
    ["observations", 0, 0],
    [],
    [],
    lambda x: relative_strength_index(x, Window(0, 0)),
)
add_case(
    "single-obs1-r1",
    "boundaries-extra",
    ["observations", 1, 1],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x, Window(1, 1)),
)
add_case(
    "single-default",
    "boundaries-extra",
    ["default", 0, 0],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x),
    rust_error="Ramp",
)
add_case(
    "single-obs2-r2",
    "boundaries-extra",
    ["observations", 2, 2],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x, Window(2, 2)),
    rust_error="Ramp",
)
add_case(
    "single-obs0",
    "boundaries-extra",
    ["observations", 0, 0],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x, Window(0, 0)),
    rust_error="Window",
)
add_case(
    "single-3d",
    "boundaries-extra",
    ["days", 3, 3],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x, "3d"),
    rust_error="Ramp",
)
add_case(
    "single-3d-r0",
    "boundaries-extra",
    ["days", 3, 0],
    JAN10_DATES[:1],
    [5.0],
    lambda x: relative_strength_index(x, Window("3d", 0)),
)
add_case(
    "two-obs2-r0",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN10_DATES[:2],
    [5.0, 6.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)
add_case(
    "two-obs2-r2",
    "boundaries-extra",
    ["observations", 2, 2],
    JAN10_DATES[:2],
    [5.0, 6.0],
    lambda x: relative_strength_index(x, Window(2, 2)),
    rust_error="Ramp",
)
add_case(
    "oversize-size-r0",
    "boundaries-extra",
    ["observations", 20, 0],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window(20, 0)),
)
add_case(
    "ramp-eq-len-gains",
    "boundaries-extra",
    ["observations", 3, 9],
    JAN10_DATES,
    MIXED_VALUES,
    lambda x: relative_strength_index(x, Window(3, 9)),
)
add_case(
    "neg-zero-const",
    "boundaries-extra",
    ["observations", 2, 0],
    JAN6_DATES[:4],
    [-0.0, -0.0, -0.0, -0.0],
    lambda x: relative_strength_index(x, Window(2, 0)),
)

payload = {
    "schema": "gsquant-rsi-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/technicals.py:relative_strength_index",
        "gs_quant/timeseries/technicals.py:smoothed_moving_average",
        "gs_quant/timeseries/helper.py:normalize_window",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "rsi-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream rsi cases")
