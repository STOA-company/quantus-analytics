"""Generate upstream prices parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries.econometrics import Returns, prices, returns

ROOT = pathlib.Path(__file__).resolve().parent

TEST_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 3),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 5),
    dt.date(2019, 1, 6),
]

JAN5_DATES = [dt.date(2020, 1, day) for day in range(1, 6)]
JAN4_DATES = JAN5_DATES[:4]
JAN3_DATES = JAN5_DATES[:3]
JAN2_DATES = JAN5_DATES[:2]
JAN1_DATES = JAN5_DATES[:1]

LEAP_DATES_4 = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 2),
]

LEAP_DATES_5 = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 10),
]

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
        "operation": "prices",
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


# 6 valid assertions from upstream test_econometrics.py::test_prices
# (invalid string type "None" MqValueError is unrepresentable with Rust Option<ReturnKind>)
add_case(
    "test-prices-empty-default",
    "test_econometrics.py::test_prices",
    ["default", "default"],
    [],
    [],
    lambda x: prices(x),
)
add_case(
    "test-prices-simple-default",
    "test_econometrics.py::test_prices",
    ["default", "default"],
    TEST_DATES,
    [NAN, 0.01, 0.02, -0.02, 0.0, 0.02],
    lambda x: prices(x),
)
add_case(
    "test-prices-simple-init100-default-kind",
    "test_econometrics.py::test_prices",
    ["100.0", "default"],
    TEST_DATES,
    [NAN, 0.01, 0.02, -0.02, 0.0, 0.02],
    lambda x: prices(x, 100),
)
add_case(
    "test-prices-simple-init100-explicit",
    "test_econometrics.py::test_prices",
    ["100.0", "simple"],
    TEST_DATES,
    [NAN, 0.01, 0.02, -0.02, 0.0, 0.02],
    lambda x: prices(x, 100, Returns.SIMPLE),
)
add_case(
    "test-prices-log-init100",
    "test_econometrics.py::test_prices",
    ["100.0", "logarithmic"],
    TEST_DATES,
    [NAN, 0.009950, 0.019803, -0.020203, 0.0, 0.019803],
    lambda x: prices(x, 100, Returns.LOGARITHMIC),
)
add_case(
    "test-prices-abs-init100",
    "test_econometrics.py::test_prices",
    ["100.0", "absolute"],
    TEST_DATES,
    [NAN, 1.0, 2.02, -2.0604, 0.0, 2.019192],
    lambda x: prices(x, 100, Returns.ABSOLUTE),
)

# Bundled synthetic boundaries from pinned-boundaries-1.json (remaining 29 distinct cases;
# simple_default and simple_init100 are identical to test-prices-simple-{default,init100-explicit})
add_case(
    "log-init100-pinned",
    "pinned-boundaries-1.json",
    ["100.0", "logarithmic"],
    TEST_DATES,
    [NAN, 0.01, 0.02, -0.02, 0.0, 0.02],
    lambda x: prices(x, 100.0, Returns.LOGARITHMIC),
)
add_case(
    "abs-init100-pinned",
    "pinned-boundaries-1.json",
    ["100.0", "absolute"],
    TEST_DATES,
    [NAN, 1.0, 2.0, -2.0, 0.0, 2.0],
    lambda x: prices(x, 100.0, Returns.ABSOLUTE),
)
add_case(
    "simple-no-nan",
    "pinned-boundaries-1.json",
    ["10.0", "simple"],
    JAN4_DATES,
    [0.1, -0.1, 0.05, 0.0],
    lambda x: prices(x, 10.0, Returns.SIMPLE),
)
add_case(
    "log-no-nan",
    "pinned-boundaries-1.json",
    ["10.0", "logarithmic"],
    JAN4_DATES,
    [0.1, -0.1, 0.05, 0.0],
    lambda x: prices(x, 10.0, Returns.LOGARITHMIC),
)
add_case(
    "abs-no-nan",
    "pinned-boundaries-1.json",
    ["10.0", "absolute"],
    JAN4_DATES,
    [1.5, -0.5, 2.0, 0.0],
    lambda x: prices(x, 10.0, Returns.ABSOLUTE),
)
add_case(
    "empty-simple",
    "pinned-boundaries-1.json",
    ["1.0", "simple"],
    [],
    [],
    lambda x: prices(x, 1.0, Returns.SIMPLE),
)
add_case(
    "empty-log",
    "pinned-boundaries-1.json",
    ["100.0", "logarithmic"],
    [],
    [],
    lambda x: prices(x, 100.0, Returns.LOGARITHMIC),
)
add_case(
    "empty-abs",
    "pinned-boundaries-1.json",
    ["100.0", "absolute"],
    [],
    [],
    lambda x: prices(x, 100.0, Returns.ABSOLUTE),
)
add_case(
    "singleton-nan-simple",
    "pinned-boundaries-1.json",
    ["50.0", "simple"],
    JAN1_DATES,
    [NAN],
    lambda x: prices(x, 50.0, Returns.SIMPLE),
)
add_case(
    "singleton-nan-log",
    "pinned-boundaries-1.json",
    ["50.0", "logarithmic"],
    JAN1_DATES,
    [NAN],
    lambda x: prices(x, 50.0, Returns.LOGARITHMIC),
)
add_case(
    "singleton-nan-abs",
    "pinned-boundaries-1.json",
    ["50.0", "absolute"],
    JAN1_DATES,
    [NAN],
    lambda x: prices(x, 50.0, Returns.ABSOLUTE),
)
add_case(
    "all-nan-simple",
    "pinned-boundaries-1.json",
    ["10.0", "simple"],
    JAN4_DATES,
    [NAN, NAN, NAN, NAN],
    lambda x: prices(x, 10.0, Returns.SIMPLE),
)
add_case(
    "all-nan-log",
    "pinned-boundaries-1.json",
    ["10.0", "logarithmic"],
    JAN4_DATES,
    [NAN, NAN, NAN, NAN],
    lambda x: prices(x, 10.0, Returns.LOGARITHMIC),
)
add_case(
    "all-nan-abs",
    "pinned-boundaries-1.json",
    ["10.0", "absolute"],
    JAN4_DATES,
    [NAN, NAN, NAN, NAN],
    lambda x: prices(x, 10.0, Returns.ABSOLUTE),
)
add_case(
    "leading-nans-simple",
    "pinned-boundaries-1.json",
    ["100.0", "simple"],
    JAN5_DATES,
    [NAN, NAN, 0.1, NAN, -0.2],
    lambda x: prices(x, 100.0, Returns.SIMPLE),
)
add_case(
    "leading-nans-log",
    "pinned-boundaries-1.json",
    ["100.0", "logarithmic"],
    JAN5_DATES,
    [NAN, NAN, 0.1, NAN, -0.2],
    lambda x: prices(x, 100.0, Returns.LOGARITHMIC),
)
add_case(
    "leading-nans-abs",
    "pinned-boundaries-1.json",
    ["100.0", "absolute"],
    JAN5_DATES,
    [NAN, NAN, 1.0, NAN, -2.0],
    lambda x: prices(x, 100.0, Returns.ABSOLUTE),
)
add_case(
    "simple-minus-one-collapse",
    "pinned-boundaries-1.json",
    ["100.0", "simple"],
    JAN4_DATES,
    [0.1, -1.0, 0.5, -2.0],
    lambda x: prices(x, 100.0, Returns.SIMPLE),
)
add_case(
    "simple-neg-lt-minus-one",
    "pinned-boundaries-1.json",
    ["10.0", "simple"],
    JAN4_DATES,
    [-1.5, -1.5, 0.2, 0.0],
    lambda x: prices(x, 10.0, Returns.SIMPLE),
)
add_case(
    "simple-inf-pos",
    "pinned-boundaries-1.json",
    ["10.0", "simple"],
    JAN4_DATES,
    [0.1, INF, 0.2, -1.0],
    lambda x: prices(x, 10.0, Returns.SIMPLE),
)
add_case(
    "simple-inf-neg",
    "pinned-boundaries-1.json",
    ["10.0", "simple"],
    JAN4_DATES,
    [0.1, NINF, 0.2, NAN],
    lambda x: prices(x, 10.0, Returns.SIMPLE),
)
add_case(
    "log-inf-pos",
    "pinned-boundaries-1.json",
    ["10.0", "logarithmic"],
    JAN4_DATES,
    [0.1, INF, 0.2, NINF],
    lambda x: prices(x, 10.0, Returns.LOGARITHMIC),
)
add_case(
    "log-overflow-800",
    "pinned-boundaries-1.json",
    ["10.0", "logarithmic"],
    JAN3_DATES,
    [0.1, 800.0, 0.2],
    lambda x: prices(x, 10.0, Returns.LOGARITHMIC),
    rust_error="Overflow",
)
add_case(
    "log-underflow-neg800",
    "pinned-boundaries-1.json",
    ["10.0", "logarithmic"],
    JAN3_DATES,
    [0.1, -800.0, 0.2],
    lambda x: prices(x, 10.0, Returns.LOGARITHMIC),
)
add_case(
    "abs-inf",
    "pinned-boundaries-1.json",
    ["10.0", "absolute"],
    JAN4_DATES,
    [1.0, INF, 2.0, NINF],
    lambda x: prices(x, 10.0, Returns.ABSOLUTE),
)
add_case(
    "init-zero-simple",
    "pinned-boundaries-1.json",
    ["0.0", "simple"],
    JAN3_DATES,
    [0.1, -0.2, 0.5],
    lambda x: prices(x, 0.0, Returns.SIMPLE),
)
add_case(
    "init-neg-zero-simple",
    "pinned-boundaries-1.json",
    ["-0.0", "simple"],
    JAN3_DATES,
    [0.1, -0.2, 0.5],
    lambda x: prices(x, -0.0, Returns.SIMPLE),
)
add_case(
    "init-neg-zero-abs",
    "pinned-boundaries-1.json",
    ["-0.0", "absolute"],
    JAN3_DATES,
    [-0.0, -0.0, 0.0],
    lambda x: prices(x, -0.0, Returns.ABSOLUTE),
)
add_case(
    "init-negative-simple",
    "pinned-boundaries-1.json",
    ["-10.0", "simple"],
    JAN3_DATES,
    [0.1, -0.2, 0.5],
    lambda x: prices(x, -10.0, Returns.SIMPLE),
)
add_case(
    "init-nan-simple",
    "pinned-boundaries-1.json",
    ["NaN", "simple"],
    JAN3_DATES,
    [0.1, -0.2, 0.5],
    lambda x: prices(x, NAN, Returns.SIMPLE),
)
add_case(
    "init-inf-simple",
    "pinned-boundaries-1.json",
    ["inf", "simple"],
    JAN3_DATES,
    [0.1, -1.0, 0.2],
    lambda x: prices(x, INF, Returns.SIMPLE),
)
add_case(
    "init-nan-empty",
    "pinned-boundaries-1.json",
    ["NaN", "simple"],
    [],
    [],
    lambda x: prices(x, NAN, Returns.SIMPLE),
)
add_case(
    "irregular-leap",
    "pinned-boundaries-1.json",
    ["100.0", "simple"],
    LEAP_DATES_4,
    [NAN, 0.05, -0.02, 0.03],
    lambda x: prices(x, 100.0, Returns.SIMPLE),
)

# Additional boundary probes: cumulative product overflow, near-max exp, signed-zero empty prefix,
# negative-Inf initial, and returns -> prices round-trips on irregular leap-day calendars.
add_case(
    "log-cumprod-overflow",
    "boundaries-extra",
    ["1.0", "logarithmic"],
    JAN2_DATES,
    [709.0, 709.0],
    lambda x: prices(x, 1.0, Returns.LOGARITHMIC),
)
add_case(
    "log-cumprod-overflow-then-underflow",
    "boundaries-extra",
    ["1.0", "logarithmic"],
    JAN5_DATES,
    [709.0, 709.0, NAN, INF, -800.0],
    lambda x: prices(x, 1.0, Returns.LOGARITHMIC),
)
add_case(
    "simple-cumprod-overflow",
    "boundaries-extra",
    ["1.0", "simple"],
    JAN5_DATES,
    [1e200, 1e200, NAN, INF, -1.0],
    lambda x: prices(x, 1.0, Returns.SIMPLE),
)
add_case(
    "log-overflow-after-pos-inf",
    "boundaries-extra",
    ["1.0", "logarithmic"],
    JAN2_DATES,
    [INF, 800.0],
    lambda x: prices(x, 1.0, Returns.LOGARITHMIC),
    rust_error="Overflow",
)
add_case(
    "log-overflow-nan-initial",
    "boundaries-extra",
    ["NaN", "logarithmic"],
    JAN1_DATES,
    [800.0],
    lambda x: prices(x, NAN, Returns.LOGARITHMIC),
    rust_error="Overflow",
)
add_case(
    "log-near-max-finite",
    "boundaries-extra",
    ["1.0", "logarithmic"],
    JAN1_DATES,
    [709.78271289338397],
    lambda x: prices(x, 1.0, Returns.LOGARITHMIC),
)
add_case(
    "log-just-over-max-overflow",
    "boundaries-extra",
    ["1.0", "logarithmic"],
    JAN1_DATES,
    [709.7827128933841],
    lambda x: prices(x, 1.0, Returns.LOGARITHMIC),
    rust_error="Overflow",
)
add_case(
    "abs-all-nan-init-neg-zero",
    "boundaries-extra",
    ["-0.0", "absolute"],
    JAN2_DATES,
    [NAN, NAN],
    lambda x: prices(x, -0.0, Returns.ABSOLUTE),
)
add_case(
    "abs-leading-nan-neg-zero",
    "boundaries-extra",
    ["-0.0", "absolute"],
    JAN3_DATES,
    [NAN, -0.0, 0.0],
    lambda x: prices(x, -0.0, Returns.ABSOLUTE),
)
add_case(
    "abs-init-neg-inf",
    "boundaries-extra",
    ["-inf", "absolute"],
    JAN3_DATES,
    [1.0, -2.0, NAN],
    lambda x: prices(x, NINF, Returns.ABSOLUTE),
)
add_case(
    "log-init-neg-zero",
    "boundaries-extra",
    ["-0.0", "logarithmic"],
    JAN3_DATES,
    [0.1, -0.2, 0.5],
    lambda x: prices(x, -0.0, Returns.LOGARITHMIC),
)
add_case(
    "log-init-neg-inf-underflow",
    "boundaries-extra",
    ["-inf", "logarithmic"],
    JAN3_DATES,
    [0.1, -800.0, 0.2],
    lambda x: prices(x, NINF, Returns.LOGARITHMIC),
)

ROUNDTRIP_PRICES = pd.Series([100.0, 105.0, 102.9, 108.0, 99.5], index=LEAP_DATES_5, dtype=float)
for mode_name, kind_enum in [
    ("simple", Returns.SIMPLE),
    ("logarithmic", Returns.LOGARITHMIC),
    ("absolute", Returns.ABSOLUTE),
]:
    r_series = returns(ROUNDTRIP_PRICES, 1, kind_enum)
    add_case(
        f"roundtrip-leap-{mode_name}",
        "boundaries-extra",
        ["100.0", mode_name],
        LEAP_DATES_5,
        [float(v) for v in r_series],
        lambda x, k=kind_enum: prices(x, 100.0, k),
    )

payload = {
    "schema": "gsquant-prices-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/econometrics.py:prices",
        "gs_quant/timeseries/statistics.py:product",
        "gs_quant/timeseries/statistics.py:sum_",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "prices-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream prices cases")
