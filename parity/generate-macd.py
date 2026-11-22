"""Generate upstream macd parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries.technicals import macd

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

JAN10_DATES = [dt.date(2020, 1, day) for day in range(1, 11)]
JAN7_DATES = JAN10_DATES[:7]
JAN6_DATES = JAN10_DATES[:6]
JAN5_DATES = JAN10_DATES[:5]
JAN4_DATES = JAN10_DATES[:4]
JAN3_DATES = JAN10_DATES[:3]
JAN1_DATES = JAN10_DATES[:1]

DEFAULT_SERIES_VALUES = [
    100.0,
    102.0,
    101.0,
    105.0,
    104.0,
    108.0,
    107.0,
    111.0,
    110.0,
    115.0,
]

# Irregular dates including leap day 2020-02-29
LEAP_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 10),
]
LEAP_VALUES = [10.0, 12.0, 11.0, 15.0, 13.0]

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


cases = []


def add_case(case_id, source_ref, args, dates, values, fn, rust_error=None):
    if dates:
        series = pd.Series(values, index=dates, dtype=float)
    else:
        series = pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "macd",
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


# 4 calls covering all 3 assertions of upstream test_technicals.py::test_macd
add_case(
    "test-macd-equal-10-10",
    "test_technicals.py::test_macd",
    ["10", "10", "default"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 10, 10),
)
add_case(
    "test-macd-10-12-default-s",
    "test_technicals.py::test_macd",
    ["10", "12", "default"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 10, 12),
)
add_case(
    "test-macd-10-12-explicit-s1",
    "test_technicals.py::test_macd",
    ["10", "12", "1"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 10, 12, s=1),
)
add_case(
    "test-macd-2-3",
    "test_technicals.py::test_macd",
    ["2", "3", "default"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 2, 3),
)

# Bundled synthetic boundaries from pinned-boundaries-1.json (33 Rust-representable cases)
add_case(
    "defaults-12-26-1",
    "pinned-boundaries-1.json",
    ["default", "default", "default"],
    JAN10_DATES,
    DEFAULT_SERIES_VALUES,
    lambda x: macd(x),
)
add_case(
    "signal-smoothing-s9",
    "pinned-boundaries-1.json",
    ["12", "26", "9"],
    JAN10_DATES,
    DEFAULT_SERIES_VALUES,
    lambda x: macd(x, 12, 26, 9),
)
add_case(
    "signal-smoothing-2-3-4",
    "pinned-boundaries-1.json",
    ["2", "3", "4"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 2, 3, 4),
)
add_case(
    "inverted-spans-m-gt-n",
    "pinned-boundaries-1.json",
    ["5", "2", "3"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 5, 2, 3),
)
add_case(
    "all-span-1",
    "pinned-boundaries-1.json",
    ["1", "1", "1"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 1, 1, 1),
)
add_case(
    "m1-n3-s2",
    "pinned-boundaries-1.json",
    ["1", "3", "2"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 1, 3, 2),
)

# Empty inputs and zero-span validation (empty and non-empty)
add_case(
    "empty-defaults",
    "pinned-boundaries-1.json",
    ["default", "default", "default"],
    [],
    [],
    lambda x: macd(x),
)
add_case(
    "empty-m0",
    "pinned-boundaries-1.json",
    ["0", "26", "1"],
    [],
    [],
    lambda x: macd(x, 0, 26, 1),
    rust_error="Span",
)
add_case(
    "empty-n0",
    "pinned-boundaries-1.json",
    ["12", "0", "1"],
    [],
    [],
    lambda x: macd(x, 12, 0, 1),
    rust_error="Span",
)
add_case(
    "empty-s0",
    "pinned-boundaries-1.json",
    ["12", "26", "0"],
    [],
    [],
    lambda x: macd(x, 12, 26, 0),
    rust_error="Span",
)
add_case(
    "nonempty-m0",
    "pinned-boundaries-1.json",
    ["0", "2", "1"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 0, 2, 1),
    rust_error="Span",
)
add_case(
    "nonempty-n0",
    "pinned-boundaries-1.json",
    ["2", "0", "1"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 2, 0, 1),
    rust_error="Span",
)
add_case(
    "nonempty-s0",
    "pinned-boundaries-1.json",
    ["2", "3", "0"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 2, 3, 0),
    rust_error="Span",
)

# Singletons and all-NaN / all-Inf
add_case(
    "singleton-finite",
    "pinned-boundaries-1.json",
    ["12", "26", "9"],
    JAN1_DATES,
    [42.0],
    lambda x: macd(x, 12, 26, 9),
)
add_case(
    "singleton-nan",
    "pinned-boundaries-1.json",
    ["2", "3", "2"],
    JAN1_DATES,
    [NAN],
    lambda x: macd(x, 2, 3, 2),
)
add_case(
    "singleton-neg-zero",
    "pinned-boundaries-1.json",
    ["2", "3", "2"],
    JAN1_DATES,
    [-0.0],
    lambda x: macd(x, 2, 3, 2),
)
add_case(
    "singleton-inf",
    "pinned-boundaries-1.json",
    ["2", "3", "2"],
    JAN1_DATES,
    [INF],
    lambda x: macd(x, 2, 3, 2),
)
add_case(
    "all-nan-len4",
    "pinned-boundaries-1.json",
    ["2", "3", "2"],
    JAN4_DATES,
    [NAN, NAN, NAN, NAN],
    lambda x: macd(x, 2, 3, 2),
)

# Leading, internal, and trailing missing values (NaN and ±Inf)
add_case(
    "leading-nans-s1",
    "pinned-boundaries-1.json",
    ["2", "4", "1"],
    JAN6_DATES,
    [NAN, NAN, 10.0, 20.0, 15.0, 25.0],
    lambda x: macd(x, 2, 4, 1),
)
add_case(
    "leading-nans-s3",
    "pinned-boundaries-1.json",
    ["2", "4", "3"],
    JAN6_DATES,
    [NAN, NAN, 10.0, 20.0, 15.0, 25.0],
    lambda x: macd(x, 2, 4, 3),
)
add_case(
    "internal-nans-s1",
    "pinned-boundaries-1.json",
    ["2", "5", "1"],
    JAN7_DATES,
    [10.0, NAN, NAN, 20.0, NAN, 5.0, 12.0],
    lambda x: macd(x, 2, 5, 1),
)
add_case(
    "internal-nans-s3",
    "pinned-boundaries-1.json",
    ["2", "5", "3"],
    JAN7_DATES,
    [10.0, NAN, NAN, 20.0, NAN, 5.0, 12.0],
    lambda x: macd(x, 2, 5, 3),
)
add_case(
    "trailing-nans-s3",
    "pinned-boundaries-1.json",
    ["2", "4", "3"],
    JAN6_DATES,
    [10.0, 20.0, 15.0, NAN, NAN, NAN],
    lambda x: macd(x, 2, 4, 3),
)
add_case(
    "leading-inf-s2",
    "pinned-boundaries-1.json",
    ["2", "4", "2"],
    JAN5_DATES,
    [INF, NINF, 10.0, 20.0, 15.0],
    lambda x: macd(x, 2, 4, 2),
)
add_case(
    "internal-inf-s2",
    "pinned-boundaries-1.json",
    ["2", "4", "2"],
    JAN6_DATES,
    [10.0, INF, 20.0, NINF, 5.0, 12.0],
    lambda x: macd(x, 2, 4, 2),
)
add_case(
    "all-inf",
    "pinned-boundaries-1.json",
    ["2", "3", "2"],
    JAN3_DATES,
    [INF, NINF, INF],
    lambda x: macd(x, 2, 3, 2),
)

# Constants, signed zeros, irregular leap day, negative values, large magnitudes
add_case(
    "constant-positive",
    "pinned-boundaries-1.json",
    ["2", "5", "3"],
    JAN6_DATES,
    [5.0] * 6,
    lambda x: macd(x, 2, 5, 3),
)
add_case(
    "constant-neg-zero",
    "pinned-boundaries-1.json",
    ["2", "5", "3"],
    JAN5_DATES,
    [-0.0] * 5,
    lambda x: macd(x, 2, 5, 3),
)
add_case(
    "mixed-signed-zeros",
    "pinned-boundaries-1.json",
    ["2", "5", "3"],
    JAN5_DATES,
    [-0.0, 0.0, -0.0, 0.0, -0.0],
    lambda x: macd(x, 2, 5, 3),
)
add_case(
    "irregular-leap-dates",
    "pinned-boundaries-1.json",
    ["2", "4", "3"],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: macd(x, 2, 4, 3),
)
add_case(
    "negative-values",
    "pinned-boundaries-1.json",
    ["3", "6", "2"],
    JAN6_DATES,
    [-10.0, -5.0, -8.0, 0.0, 4.0, -2.0],
    lambda x: macd(x, 3, 6, 2),
)
add_case(
    "large-overflow-diff",
    "pinned-boundaries-1.json",
    ["2", "10", "2"],
    JAN5_DATES,
    [1e308, -1e308, 1e308, -1e308, 1e308],
    lambda x: macd(x, 2, 10, 2),
)
add_case(
    "large-same-sign",
    "pinned-boundaries-1.json",
    ["2", "10", "2"],
    JAN5_DATES,
    [1e308, 1e308, 0.5e308, 1e308, 0.5e308],
    lambda x: macd(x, 2, 10, 2),
)

# Additional boundary probes: true a - b overflow to ±Inf entering 3rd EWM & extended edges
add_case(
    "true-overflow-diff-s1",
    "boundaries-extra",
    ["1", "20", "1"],
    JAN5_DATES,
    [1.7e308, -1.7e308, 1.7e308, -1.7e308, 1.7e308],
    lambda x: macd(x, 1, 20, 1),
)
add_case(
    "true-overflow-diff-s3",
    "boundaries-extra",
    ["1", "20", "3"],
    JAN5_DATES,
    [1.7e308, -1.7e308, 1.7e308, -1.7e308, 1.7e308],
    lambda x: macd(x, 1, 20, 3),
)
add_case(
    "true-overflow-pos-s2",
    "boundaries-extra",
    ["1", "20", "2"],
    JAN5_DATES,
    [-1.7e308, 1.7e308, -1.7e308, 1.7e308, -1.7e308],
    lambda x: macd(x, 1, 20, 2),
)
add_case(
    "leading-nan-true-overflow",
    "boundaries-extra",
    ["1", "20", "2"],
    JAN5_DATES,
    [NAN, 1.7e308, -1.7e308, 0.0, 1.0],
    lambda x: macd(x, 1, 20, 2),
)
add_case(
    "all-mixed-nonfinite",
    "boundaries-extra",
    ["2", "4", "3"],
    JAN4_DATES,
    [NAN, INF, NINF, NAN],
    lambda x: macd(x, 2, 4, 3),
)
add_case(
    "leading-nan-neg-zero-s3",
    "boundaries-extra",
    ["2", "4", "3"],
    JAN5_DATES,
    [NAN, NAN, -0.0, 2.0, 1.0],
    lambda x: macd(x, 2, 4, 3),
)
add_case(
    "trailing-inf-s3",
    "boundaries-extra",
    ["2", "4", "3"],
    JAN5_DATES,
    [10.0, 20.0, 15.0, INF, NINF],
    lambda x: macd(x, 2, 4, 3),
)
add_case(
    "m-gt-n-s1",
    "boundaries-extra",
    ["26", "12", "1"],
    JAN10_DATES,
    DEFAULT_SERIES_VALUES,
    lambda x: macd(x, 26, 12, 1),
)
add_case(
    "n1-m3-s2",
    "boundaries-extra",
    ["3", "1", "2"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: macd(x, 3, 1, 2),
)
add_case(
    "singleton-ninf",
    "boundaries-extra",
    ["2", "3", "2"],
    JAN1_DATES,
    [NINF],
    lambda x: macd(x, 2, 3, 2),
)
add_case(
    "empty-explicit-valid",
    "boundaries-extra",
    ["2", "3", "4"],
    [],
    [],
    lambda x: macd(x, 2, 3, 4),
)
add_case(
    "irregular-leap-default",
    "boundaries-extra",
    ["default", "default", "default"],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: macd(x),
)

payload = {
    "schema": "gsquant-macd-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/technicals.py:macd",
        "gs_quant/timeseries/algebra.py:subtract",
        "gs_quant/timeseries/datetime.py:align",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "macd-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream macd cases")
