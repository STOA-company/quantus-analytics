"""Generate upstream smoothed_moving_average parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries import Window, smoothed_moving_average

ROOT = pathlib.Path(__file__).resolve().parent

TEST_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 3),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 5),
    dt.date(2019, 1, 6),
]
ORIGINAL_VALUES = [3.0, 2.0, 3.0, 1.0, 3.0, 6.0]

IRREGULAR_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 7),
    dt.date(2020, 3, 14),
    dt.date(2020, 3, 23),
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
    if isinstance(d, dt.date):
        return d.isoformat()
    return str(d)


def index_kind(idx) -> str:
    if len(idx) > 0 and isinstance(idx[0], dt.date) and not isinstance(idx[0], dt.datetime):
        return "Index[date]"
    return type(idx).__name__


cases = []


def add_case(case_id, source_ref, args, dates, values, fn, rust_error=None):
    # Upstream smoothed_moving_average compares (x.index[i] - window_size).date() in calendar
    # windows, which requires a datetime.date index (DatetimeIndex raises TypeError).
    if dates:
        series = pd.Series(values, index=dates, dtype=float)
    else:
        series = pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "smoothed_moving_average",
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
                "unit": "price",
                "dtype": str(out.dtype),
            }
    cases.append(entry)


# 5 cases from upstream test_technicals.py::test_smoothed_moving_average
add_case(
    "test-smma-default",
    "test_technicals.py::test_smoothed_moving_average",
    ["expanding", 0, 0],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x),
)
add_case(
    "test-smma-obs-1",
    "test_technicals.py::test_smoothed_moving_average",
    ["observations", 1, 0],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(1, 0)),
)
add_case(
    "test-smma-obs-2",
    "test_technicals.py::test_smoothed_moving_average",
    ["observations", 2, 0],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "test-smma-2d",
    "test_technicals.py::test_smoothed_moving_average",
    ["days", 2, 2],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, "2d"),
)
# Starting 2019-01-01, DateOffset(months=1) ends at 2019-02-01 (31 calendar days), yielding empty.
add_case(
    "test-smma-1m-empty",
    "test_technicals.py::test_smoothed_moving_average",
    ["days", 31, 31],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, "1m"),
)

# Regular-date observation & calendar ramp boundaries from pinned-boundaries-1.txt
add_case(
    "reg-obs2-r2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 2],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(2, 2)),
)
add_case(
    "reg-obs3-r2",
    "pinned-boundaries-1.txt",
    ["observations", 3, 2],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(3, 2)),
)
add_case(
    "reg-2d-r0",
    "pinned-boundaries-1.txt",
    ["days", 2, 0],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("2d", 0)),
)
add_case(
    "reg-2d-r1d",
    "pinned-boundaries-1.txt",
    ["days", 2, 1],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("2d", "1d")),
)

# Irregular leap-year dates (crossing 2020-02-29) from pinned-boundaries-1.txt
add_case(
    "irreg-default",
    "pinned-boundaries-1.txt",
    ["expanding", 0, 0],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x),
)
add_case(
    "irreg-obs2-r0",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "irreg-obs2-r2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 2],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(2, 2)),
)
add_case(
    "irreg-4d",
    "pinned-boundaries-1.txt",
    ["days", 4, 4],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, "4d"),
)
add_case(
    "irreg-4d-r0",
    "pinned-boundaries-1.txt",
    ["days", 4, 0],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("4d", 0)),
)
add_case(
    "irreg-2d-gap",
    "pinned-boundaries-1.txt",
    ["days", 2, 2],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, "2d"),
)

# NaN and +Inf propagation from pinned-boundaries-1.txt
add_case(
    "nan-lead-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    TEST_DATES,
    [float("nan"), 2.0, 3.0, 1.0, 3.0, 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "nan-mid-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    TEST_DATES,
    [2.0, 4.0, float("nan"), 1.0, 3.0, 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "nan-all-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    TEST_DATES,
    [float("nan")] * 6,
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "inf-mid-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    TEST_DATES,
    [2.0, 4.0, float("inf"), 1.0, 3.0, 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)

# Empty, singleton, oversize, zero-window, and signed-zero cases from pinned-boundaries-1.txt
add_case(
    "empty-default",
    "pinned-boundaries-1.txt",
    ["expanding", 0, 0],
    [],
    [],
    lambda x: smoothed_moving_average(x),
)
add_case(
    "empty-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    [],
    [],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "empty-2d",
    "pinned-boundaries-1.txt",
    ["days", 2, 2],
    [],
    [],
    lambda x: smoothed_moving_average(x, "2d"),
    rust_error="Ramp",
)
add_case(
    "empty-2d-r0",
    "pinned-boundaries-1.txt",
    ["days", 2, 0],
    [],
    [],
    lambda x: smoothed_moving_average(x, Window("2d", 0)),
)
add_case(
    "single-default",
    "pinned-boundaries-1.txt",
    ["expanding", 0, 0],
    TEST_DATES[:1],
    [5.0],
    lambda x: smoothed_moving_average(x),
)
add_case(
    "single-obs2-r0",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    TEST_DATES[:1],
    [5.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "single-obs2-r2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 2],
    TEST_DATES[:1],
    [5.0],
    lambda x: smoothed_moving_average(x, Window(2, 2)),
    rust_error="Ramp",
)
add_case(
    "single-2d",
    "pinned-boundaries-1.txt",
    ["days", 2, 2],
    TEST_DATES[:1],
    [5.0],
    lambda x: smoothed_moving_average(x, "2d"),
)
add_case(
    "oversize-obs10-r0",
    "pinned-boundaries-1.txt",
    ["observations", 10, 0],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(10, 0)),
)
add_case(
    "oversize-obs10-r10",
    "pinned-boundaries-1.txt",
    ["observations", 10, 10],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(10, 10)),
    rust_error="Ramp",
)
add_case(
    "oversize-100d",
    "pinned-boundaries-1.txt",
    ["days", 100, 100],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, "100d"),
)
add_case(
    "zero-obs",
    "pinned-boundaries-1.txt",
    ["observations", 0, 0],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(0, 0)),
    rust_error="Window",
)
add_case(
    "neg-zero-obs2",
    "pinned-boundaries-1.txt",
    ["observations", 2, 0],
    TEST_DATES[:4],
    [-0.0, 0.0, -0.0, -0.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)

# Additional boundary probes (post-ramp calendar window counting, initial-mean NaN/Inf, signed zeros)
add_case(
    "reg-4d-r2d",
    "boundaries-extra",
    ["days", 4, 2],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("4d", "2d")),
)
add_case(
    "irreg-2d-r0",
    "boundaries-extra",
    ["days", 2, 0],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("2d", 0)),
)
add_case(
    "irreg-2d-r1d",
    "boundaries-extra",
    ["days", 2, 1],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("2d", "1d")),
)
add_case(
    "irreg-4d-r1d",
    "boundaries-extra",
    ["days", 4, 1],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("4d", "1d")),
)
add_case(
    "irreg-4d-r2d",
    "boundaries-extra",
    ["days", 4, 2],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("4d", "2d")),
)
add_case(
    "nan-in-init-obs3-r2",
    "boundaries-extra",
    ["observations", 3, 2],
    TEST_DATES[:4],
    [float("nan"), 2.0, 4.0, 6.0],
    lambda x: smoothed_moving_average(x, Window(3, 2)),
)
add_case(
    "nan-all-init-obs2-r2",
    "boundaries-extra",
    ["observations", 2, 2],
    TEST_DATES[:4],
    [float("nan"), float("nan"), 4.0, 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 2)),
)
add_case(
    "inf-lead-obs2-r0",
    "boundaries-extra",
    ["observations", 2, 0],
    TEST_DATES[:4],
    [float("inf"), 2.0, 4.0, 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "inf-lead-obs2-r1",
    "boundaries-extra",
    ["observations", 2, 1],
    TEST_DATES[:4],
    [float("inf"), 2.0, 4.0, 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 1)),
)
add_case(
    "inf-lead-2d-r0",
    "boundaries-extra",
    ["days", 2, 0],
    TEST_DATES[:4],
    [float("inf"), 2.0, 4.0, 6.0],
    lambda x: smoothed_moving_average(x, Window("2d", 0)),
)
add_case(
    "inf-lead-2d-r1d",
    "boundaries-extra",
    ["days", 2, 1],
    TEST_DATES[:4],
    [float("inf"), 2.0, 4.0, 6.0],
    lambda x: smoothed_moving_average(x, Window("2d", "1d")),
)
add_case(
    "inf-mid-2d-r0",
    "boundaries-extra",
    ["days", 2, 0],
    TEST_DATES[:4],
    [2.0, 4.0, float("inf"), 6.0],
    lambda x: smoothed_moving_average(x, Window("2d", 0)),
)
add_case(
    "inf-neg-mid-obs2-r0",
    "boundaries-extra",
    ["observations", 2, 0],
    TEST_DATES[:4],
    [2.0, 4.0, float("-inf"), 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "inf-then-ninf-obs2-r0",
    "boundaries-extra",
    ["observations", 2, 0],
    TEST_DATES[:4],
    [2.0, float("inf"), float("-inf"), 6.0],
    lambda x: smoothed_moving_average(x, Window(2, 0)),
)
add_case(
    "inf-obs1-r0",
    "boundaries-extra",
    ["observations", 1, 0],
    TEST_DATES[:3],
    [2.0, float("inf"), 6.0],
    lambda x: smoothed_moving_average(x, Window(1, 0)),
)
add_case(
    "neg-zero-obs1",
    "boundaries-extra",
    ["observations", 1, 0],
    TEST_DATES[:4],
    [-0.0, -0.0, 0.0, -0.0],
    lambda x: smoothed_moving_average(x, Window(1, 0)),
)
add_case(
    "neg-zero-obs2-r2",
    "boundaries-extra",
    ["observations", 2, 2],
    TEST_DATES[:4],
    [-0.0, -0.0, -0.0, -0.0],
    lambda x: smoothed_moving_average(x, Window(2, 2)),
)
add_case(
    "neg-zero-2d-r0",
    "boundaries-extra",
    ["days", 2, 0],
    TEST_DATES[:4],
    [-0.0, 0.0, -0.0, -0.0],
    lambda x: smoothed_moving_average(x, Window("2d", 0)),
)
add_case(
    "neg-zero-expanding",
    "boundaries-extra",
    ["expanding", 0, 0],
    TEST_DATES[:3],
    [-0.0, -0.0, -0.0],
    lambda x: smoothed_moving_average(x),
)
add_case(
    "ramp-eq-len-obs",
    "boundaries-extra",
    ["observations", 2, 6],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window(2, 6)),
)
add_case(
    "ramp-cuts-all-days",
    "boundaries-extra",
    ["days", 2, 10],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: smoothed_moving_average(x, Window("2d", "10d")),
)
add_case(
    "single-2d-r0",
    "boundaries-extra",
    ["days", 2, 0],
    TEST_DATES[:1],
    [5.0],
    lambda x: smoothed_moving_average(x, Window("2d", 0)),
)
add_case(
    "empty-obs2-r2",
    "boundaries-extra",
    ["observations", 2, 2],
    [],
    [],
    lambda x: smoothed_moving_average(x, Window(2, 2)),
)

payload = {
    "schema": "gsquant-smma-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/technicals.py:smoothed_moving_average",
        "gs_quant/timeseries/statistics.py:mean",
        "gs_quant/timeseries/helper.py:normalize_window",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "smma-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream smma cases")
