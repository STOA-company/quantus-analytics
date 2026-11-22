"""Generate upstream bollinger_bands parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries.technicals import bollinger_bands

ROOT = pathlib.Path(__file__).resolve().parent

UPSTREAM_DATES = [
    dt.date(2019, 1, 1),
    dt.date(2019, 1, 2),
    dt.date(2019, 1, 3),
    dt.date(2019, 1, 4),
    dt.date(2019, 1, 5),
    dt.date(2019, 1, 6),
]
UPSTREAM_VALUES = [3.0, 2.0, 3.0, 1.0, 3.0, 6.0]

LEAP_DATES = [
    dt.date(2020, 2, 27),
    dt.date(2020, 2, 28),
    dt.date(2020, 2, 29),
    dt.date(2020, 3, 2),
    dt.date(2020, 3, 5),
]
LEAP_VALUES = [10.0, 12.0, 11.0, 15.0, 20.0]

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
        "operation": "bollinger_bands",
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
        if rust_error is not None:
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
            lower_col = out[0]
            upper_col = out[1]
            entry["output"] = {
                "index_kind": index_kind(out.index),
                "timezone": None,
                "dates": [fmt_date(d) for d in out.index],
                "lower": [pack(float(v)) for v in lower_col],
                "upper": [pack(float(v)) for v in upper_col],
                "unit": "input",
                "dtypes": [str(lower_col.dtype), str(upper_col.dtype)],
            }
    cases.append(entry)


# 2 assertions from upstream test_technicals.py::test_bollinger_bands
add_case(
    "test-bb-expanding-default",
    "test_technicals.py::test_bollinger_bands",
    ["expanding", 0, 0, "default"],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x),
)
add_case(
    "test-bb-cal2d-r2d-default",
    "test_technicals.py::test_bollinger_bands",
    ["days", 2, 2, "default"],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, "2d"),
)

# Bundled synthetic boundaries from pinned-boundaries-1.json
# (upstream_default overlaps with test-bb-expanding-default above; days-size0 tagged as
# documented Rust Error::Window rejection)
add_case(
    "upstream-k15-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(1.5)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 0), 1.5),
)
add_case(
    "obs3-r2-k2",
    "pinned-boundaries-1.json",
    ["observations", 3, 2, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 2), 2.0),
)
add_case(
    "days3-r0-k2",
    "pinned-boundaries-1.json",
    ["days", 3, 0, pack(2.0)],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: bollinger_bands(x, Window("3d", 0), 2.0),
)
add_case(
    "days3-r3-k2",
    "pinned-boundaries-1.json",
    ["days", 3, 3, pack(2.0)],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: bollinger_bands(x, Window("3d", "3d"), 2.0),
)
add_case(
    "k-zero",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(0.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 0), 0.0),
)
add_case(
    "k-neg-zero",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(-0.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 0), -0.0),
)
add_case(
    "k-negative",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(-2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 0), -2.0),
)
add_case(
    "k-nan",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(NAN)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 0), NAN),
)
add_case(
    "k-inf",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(INF)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 0), INF),
)
add_case(
    "k-neg-inf",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(NINF)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(3, 0), NINF),
)
add_case(
    "empty-expanding",
    "pinned-boundaries-1.json",
    ["expanding", 0, 0, pack(2.0)],
    [],
    [],
    lambda x: bollinger_bands(x, Window(None, 0), 2.0),
)
add_case(
    "empty-obs2",
    "pinned-boundaries-1.json",
    ["observations", 2, 0, pack(2.0)],
    [],
    [],
    lambda x: bollinger_bands(x, Window(2, 0), 2.0),
)
add_case(
    "empty-obs0",
    "pinned-boundaries-1.json",
    ["observations", 0, 0, pack(2.0)],
    [],
    [],
    lambda x: bollinger_bands(x, Window(0, 0), 2.0),
)
add_case(
    "empty-days2-r2",
    "pinned-boundaries-1.json",
    ["days", 2, 2, pack(2.0)],
    [],
    [],
    lambda x: bollinger_bands(x, Window("2d", "2d"), 2.0),
)
add_case(
    "singleton-expanding",
    "pinned-boundaries-1.json",
    ["expanding", 0, 0, pack(2.0)],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: bollinger_bands(x, Window(None, 0), 2.0),
)
add_case(
    "obs-size0",
    "pinned-boundaries-1.json",
    ["observations", 0, 0, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(0, 0), 2.0),
    rust_error="Window",
)
add_case(
    "days-size0",
    "pinned-boundaries-1.json",
    ["days", 0, 0, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window("0d", 0), 2.0),
    rust_error="Window",
    upstream_note="upstream_zero_day_offset_fallthrough",
)
add_case(
    "obs-ramp-gt-len",
    "pinned-boundaries-1.json",
    ["observations", 2, 7, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(2, 7), 2.0),
    rust_error="Ramp",
)
add_case(
    "obs-ramp-eq-len",
    "pinned-boundaries-1.json",
    ["observations", 2, 6, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(2, 6), 2.0),
)
add_case(
    "obs-oversize",
    "pinned-boundaries-1.json",
    ["observations", 10, 0, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(10, 0), 2.0),
)
add_case(
    "constant-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [4.2] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), 2.0),
)
add_case(
    "constant-neg0-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [-0.0] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), 2.0),
)
add_case(
    "nan-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [3.0, NAN, 5.0, NAN, NAN, 7.0],
    lambda x: bollinger_bands(x, Window(3, 0), 2.0),
)
add_case(
    "inf-obs3",
    "pinned-boundaries-1.json",
    ["observations", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [3.0, INF, 5.0, 7.0, NINF, 9.0],
    lambda x: bollinger_bands(x, Window(3, 0), 2.0),
)
add_case(
    "inf-days4",
    "pinned-boundaries-1.json",
    ["days", 4, 0, pack(2.0)],
    UPSTREAM_DATES,
    [3.0, INF, 5.0, 7.0, NINF, 9.0],
    lambda x: bollinger_bands(x, Window("4d", 0), 2.0),
)

# Additional boundary probes: leap-day variants, constants & signed-zero k edges,
# all-NaN & mid-NaN days, expanding/partial/all ±Inf, size-1, singleton, empty, error precedence
add_case(
    "leap-7d-r0-k15",
    "boundaries-extra",
    ["days", 7, 0, pack(1.5)],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: bollinger_bands(x, Window("7d", 0), 1.5),
)
add_case(
    "leap-expanding-default",
    "boundaries-extra",
    ["expanding", 0, 0, "default"],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: bollinger_bands(x),
)
add_case(
    "leap-obs3-r1-k15",
    "boundaries-extra",
    ["observations", 3, 1, pack(1.5)],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: bollinger_bands(x, Window(3, 1), 1.5),
)
add_case(
    "days-1d-r0",
    "boundaries-extra",
    ["days", 1, 0, pack(2.0)],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: bollinger_bands(x, Window("1d", 0), 2.0),
)
add_case(
    "days-oversize-ramp",
    "boundaries-extra",
    ["days", 30, 30, pack(2.0)],
    LEAP_DATES,
    LEAP_VALUES,
    lambda x: bollinger_bands(x, Window("30d", "30d"), 2.0),
)
add_case(
    "obs1-r0",
    "boundaries-extra",
    ["observations", 1, 0, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(1, 0), 2.0),
)
add_case(
    "constant-days3",
    "boundaries-extra",
    ["days", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [4.2] * 6,
    lambda x: bollinger_bands(x, Window("3d", 0), 2.0),
)
add_case(
    "constant-neg0-days3",
    "boundaries-extra",
    ["days", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [-0.0] * 6,
    lambda x: bollinger_bands(x, Window("3d", 0), 2.0),
)
add_case(
    "constant-neg0-k-zero",
    "boundaries-extra",
    ["observations", 3, 0, pack(0.0)],
    UPSTREAM_DATES,
    [-0.0] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), 0.0),
)
add_case(
    "constant-neg0-k-neg-zero",
    "boundaries-extra",
    ["observations", 3, 0, pack(-0.0)],
    UPSTREAM_DATES,
    [-0.0] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), -0.0),
)
add_case(
    "constant-neg0-k-negative",
    "boundaries-extra",
    ["observations", 3, 0, pack(-2.0)],
    UPSTREAM_DATES,
    [-0.0] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), -2.0),
)
add_case(
    "constant-k-inf",
    "boundaries-extra",
    ["observations", 3, 0, pack(INF)],
    UPSTREAM_DATES,
    [4.2] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), INF),
)
add_case(
    "constant-k-neg-inf",
    "boundaries-extra",
    ["observations", 3, 0, pack(NINF)],
    UPSTREAM_DATES,
    [4.2] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), NINF),
)
add_case(
    "nan-days4",
    "boundaries-extra",
    ["days", 4, 0, pack(2.0)],
    UPSTREAM_DATES,
    [3.0, NAN, 5.0, NAN, NAN, 7.0],
    lambda x: bollinger_bands(x, Window("4d", 0), 2.0),
)
add_case(
    "all-nan-obs3",
    "boundaries-extra",
    ["observations", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [NAN] * 6,
    lambda x: bollinger_bands(x, Window(3, 0), 2.0),
)
add_case(
    "all-nan-days3",
    "boundaries-extra",
    ["days", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [NAN] * 6,
    lambda x: bollinger_bands(x, Window("3d", 0), 2.0),
)
add_case(
    "inf-expanding",
    "boundaries-extra",
    ["expanding", 0, 0, "default"],
    UPSTREAM_DATES,
    [3.0, INF, 5.0, 7.0, NINF, 9.0],
    lambda x: bollinger_bands(x),
)
add_case(
    "inf-days3-partial",
    "boundaries-extra",
    ["days", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [1.0, 3.0, INF, 5.0, 7.0, 9.0],
    lambda x: bollinger_bands(x, Window("3d", 0), 2.0),
)
add_case(
    "all-inf-obs3",
    "boundaries-extra",
    ["observations", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [INF, NINF, INF, NINF, INF, NINF],
    lambda x: bollinger_bands(x, Window(3, 0), 2.0),
)
add_case(
    "all-inf-days3",
    "boundaries-extra",
    ["days", 3, 0, pack(2.0)],
    UPSTREAM_DATES,
    [INF, NINF, INF, NINF, INF, NINF],
    lambda x: bollinger_bands(x, Window("3d", 0), 2.0),
)
add_case(
    "empty-days0-r0",
    "boundaries-extra",
    ["days", 0, 0, pack(2.0)],
    [],
    [],
    lambda x: bollinger_bands(x, Window("0d", 0), 2.0),
)
add_case(
    "empty-k-nan",
    "boundaries-extra",
    ["expanding", 0, 0, pack(NAN)],
    [],
    [],
    lambda x: bollinger_bands(x, Window(None, 0), NAN),
)
add_case(
    "singleton-obs1-r0",
    "boundaries-extra",
    ["observations", 1, 0, pack(2.0)],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: bollinger_bands(x, Window(1, 0), 2.0),
)
add_case(
    "singleton-obs2-r0",
    "boundaries-extra",
    ["observations", 2, 0, pack(2.0)],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: bollinger_bands(x, Window(2, 0), 2.0),
)
add_case(
    "singleton-days2-r0",
    "boundaries-extra",
    ["days", 2, 0, pack(2.0)],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: bollinger_bands(x, Window("2d", 0), 2.0),
)
add_case(
    "singleton-obs1-r2",
    "boundaries-extra",
    ["observations", 1, 2, pack(2.0)],
    SINGLE_DATES,
    SINGLE_VALUES,
    lambda x: bollinger_bands(x, Window(1, 2), 2.0),
    rust_error="Ramp",
)
add_case(
    "obs-oversize-and-ramp-gt-len",
    "boundaries-extra",
    ["observations", 10, 10, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(10, 10), 2.0),
    rust_error="Ramp",
)
add_case(
    "obs-size0-and-ramp-gt-len",
    "boundaries-extra",
    ["observations", 0, 10, pack(2.0)],
    UPSTREAM_DATES,
    UPSTREAM_VALUES,
    lambda x: bollinger_bands(x, Window(0, 10), 2.0),
    rust_error="Window",
)
add_case(
    "neg-values-obs3-k15",
    "boundaries-extra",
    ["observations", 3, 0, pack(1.5)],
    UPSTREAM_DATES,
    [-5.0, -2.0, -8.0, -2.0, -11.0, -5.0],
    lambda x: bollinger_bands(x, Window(3, 0), 1.5),
)

# Documented upstream parity boundaries (untouched Python outputs stored verbatim):
# 1. Mixed signed zeros [+0.0, -0.0]: pandas 2.3.3 rolling().mean() GH#42064 equal-value fast path
#    adopts the last zero's sign (-0.0), emitting (-0.0, +0.0); Rust IEEE-754 +0.0 + (-0.0) = +0.0
#    emits (+0.0, +0.0) (numeric difference 0).
add_case(
    "mixed-signed-zeros-exp",
    "boundaries-extra",
    ["expanding", 0, 0, "default"],
    UPSTREAM_DATES[:2],
    [0.0, -0.0],
    lambda x: bollinger_bands(x),
    upstream_note="pandas_rolling_mean_mixed_signed_zero_gh42064",
)
add_case(
    "mixed-signed-zeros-obs2",
    "boundaries-extra",
    ["observations", 2, 0, pack(2.0)],
    UPSTREAM_DATES[:2],
    [0.0, -0.0],
    lambda x: bollinger_bands(x, Window(2, 0), 2.0),
    upstream_note="pandas_rolling_mean_mixed_signed_zero_gh42064",
)
# 2. Constant 4.2 over 10 calendar days with k = ±Inf: NumPy 2.2.6 rolling_std retains a ~9.36e-16
#    mean-rounding residue on day 10 that amplifies to (-Inf, +Inf) / (+Inf, -Inf); Rust reuses
#    std's exact +0.0 constant-window contract so 0.0 * ±Inf = NaN.
TEN_DAYS = [dt.date(2019, 1, day) for day in range(1, 11)]
add_case(
    "constant-42-days10-k-inf",
    "boundaries-extra",
    ["days", 10, 0, pack(INF)],
    TEN_DAYS,
    [4.2] * 10,
    lambda x: bollinger_bands(x, Window("10d", 0), INF),
    upstream_note="numpy_calendar_const_std_rounding_residue_k_inf",
)
add_case(
    "constant-42-days10-k-neg-inf",
    "boundaries-extra",
    ["days", 10, 0, pack(NINF)],
    TEN_DAYS,
    [4.2] * 10,
    lambda x: bollinger_bands(x, Window("10d", 0), NINF),
    upstream_note="numpy_calendar_const_std_rounding_residue_k_inf",
)

payload = {
    "schema": "gsquant-bollinger-bands-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/technicals.py:bollinger_bands",
        "gs_quant/timeseries/technicals.py:moving_average",
        "gs_quant/timeseries/statistics.py:mean",
        "gs_quant/timeseries/statistics.py:std",
        "gs_quant/timeseries/statistics.py:rolling_std",
        "gs_quant/timeseries/helper.py:normalize_window",
        "gs_quant/timeseries/helper.py:apply_ramp",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "bollinger-bands-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream bollinger_bands cases")
