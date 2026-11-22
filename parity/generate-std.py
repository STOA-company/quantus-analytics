"""Generate upstream std parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries.statistics import std

ROOT = pathlib.Path(__file__).resolve().parent

UPSTREAM_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 3),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 7),
    dt.date(2019, 1, 8),
]
UPSTREAM_VALUES = [3.0, 2.0, 3.0, 1.0, 3.0, 6.0]

LEAP_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 5),
]
LEAP_VALUES = [1.0, 3.0, 2.0, 6.0, 10.0]

SINGLE_DATES = [dt.date(2020, 1, 1)]
SINGLE_VALUES = [5.0]

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


def add_case(
    case_id,
    source_ref,
    args,
    dates,
    values,
    fn,
    rust_error=None,
    upstream_note=None,
):
    if dates:
        series = pd.Series(values, index=dates, dtype=float)
    else:
        series = pd.Series(dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "std",
        "args": args,
        "input": {
            "index_kind": index_kind(series.index),
            "timezone": None,
            "dates": [d.isoformat() for d in dates],
            "values": [pack(v) for v in values],
            "dtype": "float64",
        },
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note
        entry["rust_error"] = rust_error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = fn(series)
        except Exception as exc:
            if rust_error is None or upstream_note is not None:
                raise
            entry["error"] = {
                "rust": rust_error,
                "upstream_type": type(exc).__name__,
                "upstream_message": str(exc),
            }
        else:
            if rust_error is not None and upstream_note is None:
                raise AssertionError(
                    f"{case_id}: expected upstream error for {rust_error}, got output"
                )
            entry["output"] = {
                "index_kind": index_kind(out.index),
                "timezone": None,
                "dates": [fmt_date(d) for d in out.index],
                "values": [pack(float(v)) for v in out],
                "unit": "input",
                "dtype": str(out.dtype),
            }
    cases.append(entry)


# 4 assertions from upstream test_statistics.py::test_std
add_case(
    "test-std-expanding",
    "test_statistics.py::test_std",
    ["expanding", 0, 0],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x),
)
add_case(
    "test-std-obs2-r0",
    "test_statistics.py::test_std",
    ["observations", 2, 0],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(2, 0)),
)
add_case(
    "test-std-cal7d-r0",
    "test_statistics.py::test_std",
    ["days", 7, 0],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window("1w", 0)),
)
add_case(
    "test-std-empty",
    "test_statistics.py::test_std",
    ["expanding", 0, 0],
    [],
    [],
    lambda x: std(x),
)

# Bundled synthetic boundaries from pinned-boundaries-1.json
# (4 overlap with test_std above; large_obs3 excluded due to documented pandas Welford
# overflow divergence; days-size0 tagged as documented Rust Window rejection)
add_case(
    "obs3-r2",
    "pinned-boundaries-1.json",
    ["observations", 3, 2],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(3, 2)),
)
add_case(
    "obs1-r0",
    "pinned-boundaries-1.json",
    ["observations", 1, 0],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(1, 0)),
)
add_case(
    "obs-oversize",
    "pinned-boundaries-1.json",
    ["observations", 10, 0],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(10, 0)),
)
add_case(
    "obs-ramp-eq-len",
    "pinned-boundaries-1.json",
    ["observations", 2, 6],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(2, 6)),
)
add_case(
    "obs-ramp-gt-len",
    "pinned-boundaries-1.json",
    ["observations", 2, 7],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(2, 7)),
    rust_error="Ramp",
)
add_case(
    "obs-size0",
    "pinned-boundaries-1.json",
    ["observations", 0, 0],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(0, 0)),
    rust_error="Window",
)
add_case(
    "days-size0",
    "pinned-boundaries-1.json",
    ["days", 0, 0],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window("0d", 0)),
    rust_error="Window",
    upstream_note="upstream_zero_day_offset_fallthrough",
)
add_case(
    "days-3d-r3d",
    "pinned-boundaries-1.json",
    ["days", 3, 3],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: std(x, Window("3d", "3d")),
)
add_case(
    "days-3d-r0",
    "pinned-boundaries-1.json",
    ["days", 3, 0],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: std(x, Window("3d", 0)),
)
add_case(
    "days-1d-r0",
    "pinned-boundaries-1.json",
    ["days", 1, 0],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: std(x, Window("1d", 0)),
)
add_case(
    "days-oversize-ramp",
    "pinned-boundaries-1.json",
    ["days", 30, 30],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: std(x, Window("30d", "30d")),
)
add_case(
    "empty-obs2",
    "pinned-boundaries-1.json",
    ["observations", 2, 0],
    [],
    [],
    lambda x: std(x, Window(2, 0)),
)
add_case(
    "empty-obs0",
    "pinned-boundaries-1.json",
    ["observations", 0, 0],
    [],
    [],
    lambda x: std(x, Window(0, 0)),
)
add_case(
    "empty-days2-r2",
    "pinned-boundaries-1.json",
    ["days", 2, 2],
    [],
    [],
    lambda x: std(x, Window("2d", "2d")),
)
add_case(
    "singleton-expanding",
    "pinned-boundaries-1.json",
    ["expanding", 0, 0],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: std(x),
)
add_case(
    "singleton-days2-r0",
    "pinned-boundaries-1.json",
    ["days", 2, 0],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: std(x, Window("2d", 0)),
)
add_case(
    "constant-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [4.2] * 6,
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "constant-days3",
    "pinned-boundaries-1.json",
    ["days", 3, 0],
    UPSTREAM_DATES,
    [4.2] * 6,
    lambda x: std(x, Window("3d", 0)),
)
add_case(
    "constant-01-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [0.1] * 6,
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "constant-01-days3",
    "pinned-boundaries-1.json",
    ["days", 3, 0],
    UPSTREAM_DATES,
    [0.1] * 6,
    lambda x: std(x, Window("3d", 0)),
)
add_case(
    "constant-neg0-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [-0.0] * 6,
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "constant-neg0-days3",
    "pinned-boundaries-1.json",
    ["days", 3, 0],
    UPSTREAM_DATES,
    [-0.0] * 6,
    lambda x: std(x, Window("3d", 0)),
)
add_case(
    "nan-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [1.0, NAN, 3.0, NAN, NAN, 5.0],
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "nan-days4",
    "pinned-boundaries-1.json",
    ["days", 4, 0],
    UPSTREAM_DATES,
    [1.0, NAN, 3.0, NAN, NAN, 5.0],
    lambda x: std(x, Window("4d", 0)),
)
add_case(
    "all-nan-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [NAN] * 6,
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "all-nan-days3",
    "pinned-boundaries-1.json",
    ["days", 3, 0],
    UPSTREAM_DATES,
    [NAN] * 6,
    lambda x: std(x, Window("3d", 0)),
)
add_case(
    "inf-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [1.0, INF, 3.0, 5.0, NINF, 7.0],
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "inf-days4",
    "pinned-boundaries-1.json",
    ["days", 4, 0],
    UPSTREAM_DATES,
    [1.0, INF, 3.0, 5.0, NINF, 7.0],
    lambda x: std(x, Window("4d", 0)),
)
add_case(
    "large-days4",
    "pinned-boundaries-1.json",
    ["days", 4, 0],
    UPSTREAM_DATES,
    [1e200, 1e200, -1e200, 1e200, -1e200, 1e200],
    lambda x: std(x, Window("4d", 0)),
)

# Additional boundary probes: leap-day 7d/expanding/obs, empty days0, singleton obs,
# error precedence, expanding ±Inf, partial ±Inf in days, negative values, mixed signed zeros, all ±Inf
add_case(
    "leap-7d-r0",
    "boundaries-extra",
    ["days", 7, 0],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: std(x, Window("7d", 0)),
)
add_case(
    "leap-expanding",
    "boundaries-extra",
    ["expanding", 0, 0],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: std(x),
)
add_case(
    "leap-obs3-r1",
    "boundaries-extra",
    ["observations", 3, 1],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: std(x, Window(3, 1)),
)
add_case(
    "empty-days0-r0",
    "boundaries-extra",
    ["days", 0, 0],
    [],
    [],
    lambda x: std(x, Window("0d", 0)),
)
add_case(
    "singleton-obs1-r0",
    "boundaries-extra",
    ["observations", 1, 0],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: std(x, Window(1, 0)),
)
add_case(
    "singleton-obs2-r0",
    "boundaries-extra",
    ["observations", 2, 0],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: std(x, Window(2, 0)),
)
add_case(
    "singleton-obs1-r1",
    "boundaries-extra",
    ["observations", 1, 1],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: std(x, Window(1, 1)),
)
add_case(
    "singleton-obs1-r2",
    "boundaries-extra",
    ["observations", 1, 2],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: std(x, Window(1, 2)),
    rust_error="Ramp",
)
add_case(
    "obs-oversize-and-ramp-gt-len",
    "boundaries-extra",
    ["observations", 10, 10],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(10, 10)),
    rust_error="Ramp",
)
add_case(
    "obs-size0-and-ramp-gt-len",
    "boundaries-extra",
    ["observations", 0, 10],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: std(x, Window(0, 10)),
    rust_error="Window",
)
add_case(
    "inf-expanding",
    "boundaries-extra",
    ["expanding", 0, 0],
    UPSTREAM_DATES,
    [1.0, INF, 3.0, 5.0, NINF, 7.0],
    lambda x: std(x),
)
add_case(
    "inf-days3-partial",
    "boundaries-extra",
    ["days", 3, 0],
    UPSTREAM_DATES,
    [1.0, 3.0, INF, 5.0, 7.0, 9.0],
    lambda x: std(x, Window("3d", 0)),
)
add_case(
    "neg-values-obs3",
    "boundaries-extra",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [-5.0, -2.0, -8.0, -2.0, -11.0, -5.0],
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "mixed-signed-zeros-obs3",
    "boundaries-extra",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "mixed-signed-zeros-days3",
    "boundaries-extra",
    ["days", 3, 0],
    UPSTREAM_DATES,
    [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
    lambda x: std(x, Window("3d", 0)),
)
add_case(
    "all-inf-obs3",
    "boundaries-extra",
    ["observations", 3, 0],
    UPSTREAM_DATES,
    [INF, NINF, INF, NINF, INF, NINF],
    lambda x: std(x, Window(3, 0)),
)
add_case(
    "all-inf-days3",
    "boundaries-extra",
    ["days", 3, 0],
    UPSTREAM_DATES,
    [INF, NINF, INF, NINF, INF, NINF],
    lambda x: std(x, Window("3d", 0)),
)

payload = {
    "schema": "gsquant-std-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/statistics.py:std",
        "gs_quant/timeseries/statistics.py:rolling_std",
        "gs_quant/timeseries/helper.py:normalize_window",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "std-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream std cases")
