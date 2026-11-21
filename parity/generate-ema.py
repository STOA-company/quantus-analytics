"""Generate upstream exponential_moving_average parity vectors only. Run with the locked upstream Python environment."""
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
from gs_quant.timeseries.technicals import exponential_moving_average

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
    return str(d)


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
    index = pd.to_datetime(dates) if dates else pd.DatetimeIndex([])
    series = pd.Series(values, index=index, dtype=float)
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "exponential_moving_average",
        "args": args,
        "input": {
            "index_kind": type(series.index).__name__,
            "timezone": None,
            "dates": [d.isoformat() for d in dates],
            "values": [pack(v) for v in values],
            "dtype": "float64",
        },
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note
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
            if rust_error is not None and upstream_note is None:
                raise AssertionError(
                    f"{case_id}: expected upstream error for {rust_error}, got output"
                )
            if rust_error is not None:
                entry["rust_error"] = rust_error
            entry["output"] = {
                "index_kind": type(out.index).__name__,
                "timezone": None,
                "dates": [fmt_date(d) for d in out.index],
                "values": [pack(float(v)) for v in out],
                "unit": "price",
                "dtype": str(out.dtype),
            }
    cases.append(entry)


# 3 cases from upstream test_technicals.py::test_exponential_moving_average
add_case(
    "test-ema-default",
    "test_technicals.py::test_exponential_moving_average",
    ["default"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: exponential_moving_average(x),
)
add_case(
    "test-ema-beta-0.6",
    "test_technicals.py::test_exponential_moving_average",
    ["0.6"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: exponential_moving_average(x, 0.6),
)
add_case(
    "test-ema-beta-0.0",
    "test_technicals.py::test_exponential_moving_average",
    ["0.0"],
    TEST_DATES,
    ORIGINAL_VALUES,
    lambda x: exponential_moving_average(x, 0.0),
)

# Irregular leap-year dates (row spacing, not calendar spacing) from pinned-boundaries-1.txt
add_case(
    "original-irregular-default",
    "pinned-boundaries-1.txt",
    ["default"],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: exponential_moving_average(x),
)
add_case(
    "original-irregular-beta-0.6",
    "pinned-boundaries-1.txt",
    ["0.6"],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: exponential_moving_average(x, 0.6),
)
add_case(
    "original-irregular-beta-0.0",
    "pinned-boundaries-1.txt",
    ["0.0"],
    IRREGULAR_DATES,
    ORIGINAL_VALUES,
    lambda x: exponential_moving_average(x, 0.0),
)

# Empty inputs (valid default, invalid 1.0, invalid -0.1, documented NaN bypass)
add_case(
    "empty-default",
    "pinned-boundaries-1.txt",
    ["default"],
    [],
    [],
    lambda x: exponential_moving_average(x),
)
add_case(
    "empty-beta-1.0",
    "pinned-boundaries-1.txt",
    ["1.0"],
    [],
    [],
    lambda x: exponential_moving_average(x, 1.0),
    rust_error="Beta",
)
add_case(
    "empty-beta-neg-0.1",
    "boundaries-extra",
    ["-0.1"],
    [],
    [],
    lambda x: exponential_moving_average(x, -0.1),
    rust_error="Beta",
)
add_case(
    "empty-beta-nan",
    "pinned-boundaries-1.txt",
    ["NaN"],
    [],
    [],
    lambda x: exponential_moving_average(x, float("nan")),
    rust_error="Beta",
    upstream_note="pandas_nan_beta_domain_bypass",
)

# Leading NaN values
add_case(
    "leading-nan-default",
    "pinned-boundaries-1.txt",
    ["default"],
    IRREGULAR_DATES[:4],
    [float("nan"), float("nan"), 2.0, 6.0],
    lambda x: exponential_moving_average(x),
)
add_case(
    "leading-nan-beta-0.0",
    "pinned-boundaries-1.txt",
    ["0.0"],
    IRREGULAR_DATES[:4],
    [float("nan"), float("nan"), 2.0, 6.0],
    lambda x: exponential_moving_average(x, 0.0),
)

# Missing gaps with valid observation after gap and trailing NaN
GAP_VALUES = [1.0, float("nan"), float("nan"), 4.0, float("nan")]
add_case(
    "gaps-default",
    "pinned-boundaries-1.txt",
    ["default"],
    IRREGULAR_DATES[:5],
    GAP_VALUES,
    lambda x: exponential_moving_average(x),
)
add_case(
    "gaps-beta-0.0",
    "pinned-boundaries-1.txt",
    ["0.0"],
    IRREGULAR_DATES[:5],
    GAP_VALUES,
    lambda x: exponential_moving_average(x, 0.0),
)
add_case(
    "gaps-beta-0.6",
    "pinned-boundaries-1.txt",
    ["0.6"],
    IRREGULAR_DATES[:5],
    GAP_VALUES,
    lambda x: exponential_moving_average(x, 0.6),
)
add_case(
    "gaps-beta-near-1",
    "pinned-boundaries-1.txt",
    ["0.9999999999999999"],
    IRREGULAR_DATES[:5],
    GAP_VALUES,
    lambda x: exponential_moving_average(x, 0.9999999999999999),
)

# Infinities treated as missing (leading +Inf and interior -Inf)
INF_VALUES = [float("inf"), 3.0, float("-inf"), 6.0]
add_case(
    "infinities-default",
    "pinned-boundaries-1.txt",
    ["default"],
    IRREGULAR_DATES[:4],
    INF_VALUES,
    lambda x: exponential_moving_average(x),
)
add_case(
    "infinities-beta-0.0",
    "pinned-boundaries-1.txt",
    ["0.0"],
    IRREGULAR_DATES[:4],
    INF_VALUES,
    lambda x: exponential_moving_average(x, 0.0),
)

# Signed-zero preservation and transitions
add_case(
    "signed-zero-neg-first-default",
    "pinned-boundaries-1.txt",
    ["default"],
    IRREGULAR_DATES[:4],
    [-0.0, 0.0, float("nan"), -0.0],
    lambda x: exponential_moving_average(x),
)
add_case(
    "signed-zero-neg-first-beta-0.0",
    "pinned-boundaries-1.txt",
    ["0.0"],
    IRREGULAR_DATES[:4],
    [-0.0, 0.0, float("nan"), -0.0],
    lambda x: exponential_moving_average(x, 0.0),
)
add_case(
    "signed-zero-pos-first-default",
    "boundaries-extra",
    ["default"],
    IRREGULAR_DATES[:4],
    [0.0, -0.0, float("nan"), 0.0],
    lambda x: exponential_moving_average(x),
)
add_case(
    "signed-zero-pos-first-beta-0.0",
    "boundaries-extra",
    ["0.0"],
    IRREGULAR_DATES[:4],
    [0.0, -0.0, float("nan"), 0.0],
    lambda x: exponential_moving_average(x, 0.0),
)
add_case(
    "signed-zero-from-pos-beta-0.0",
    "boundaries-extra",
    ["0.0"],
    IRREGULAR_DATES[:2],
    [1.0, -0.0],
    lambda x: exponential_moving_average(x, 0.0),
)
add_case(
    "signed-zero-from-neg-beta-0.0",
    "boundaries-extra",
    ["0.0"],
    IRREGULAR_DATES[:2],
    [-1.0, -0.0],
    lambda x: exponential_moving_average(x, 0.0),
)
add_case(
    "neg-zero-beta",
    "boundaries-extra",
    ["-0.0"],
    IRREGULAR_DATES[:3],
    [3.0, 2.0, 3.0],
    lambda x: exponential_moving_average(x, -0.0),
)

# Singletons and all-missing series
add_case(
    "singleton-finite-default",
    "boundaries-extra",
    ["default"],
    IRREGULAR_DATES[:1],
    [42.0],
    lambda x: exponential_moving_average(x),
)
add_case(
    "singleton-neg-zero-default",
    "boundaries-extra",
    ["default"],
    IRREGULAR_DATES[:1],
    [-0.0],
    lambda x: exponential_moving_average(x),
)
add_case(
    "singleton-nan-default",
    "boundaries-extra",
    ["default"],
    IRREGULAR_DATES[:1],
    [float("nan")],
    lambda x: exponential_moving_average(x),
)
add_case(
    "singleton-inf-default",
    "boundaries-extra",
    ["default"],
    IRREGULAR_DATES[:1],
    [float("inf")],
    lambda x: exponential_moving_average(x),
)
add_case(
    "all-nan-default",
    "boundaries-extra",
    ["default"],
    IRREGULAR_DATES[:3],
    [float("nan"), float("nan"), float("nan")],
    lambda x: exponential_moving_average(x),
)
add_case(
    "all-inf-default",
    "boundaries-extra",
    ["default"],
    IRREGULAR_DATES[:3],
    [float("inf"), float("-inf"), float("inf")],
    lambda x: exponential_moving_average(x),
)

# Invalid beta errors from pinned-boundaries-1.txt
for case_id, arg_str, beta_val in [
    ("invalid-beta-neg-0.1", "-0.1", -0.1),
    ("invalid-beta-1.0", "1.0", 1.0),
    ("invalid-beta-1.1", "1.1", 1.1),
    ("invalid-beta-pos-inf", "inf", float("inf")),
    ("invalid-beta-neg-inf", "-inf", float("-inf")),
]:
    add_case(
        case_id,
        "pinned-boundaries-1.txt",
        [arg_str],
        IRREGULAR_DATES[:1],
        [1.0],
        lambda x, b=beta_val: exponential_moving_average(x, b),
        rust_error="Beta",
    )

# Documented upstream NaN-beta bypass cases (singleton + 4 multi-point series from pinned-nan-beta.txt)
add_case(
    "invalid-beta-nan-singleton",
    "pinned-boundaries-1.txt",
    ["NaN"],
    IRREGULAR_DATES[:1],
    [1.0],
    lambda x: exponential_moving_average(x, float("nan")),
    rust_error="Beta",
    upstream_note="pandas_nan_beta_domain_bypass",
)
for case_id, vals in [
    ("nan-beta-2-3-4", [2.0, 3.0, 4.0]),
    ("nan-beta-2-2-3", [2.0, 2.0, 3.0]),
    ("nan-beta-nan-2-nan-3", [float("nan"), 2.0, float("nan"), 3.0]),
    ("nan-beta-2-3-2-2", [2.0, 3.0, 2.0, 2.0]),
]:
    add_case(
        case_id,
        "pinned-nan-beta.txt",
        ["NaN"],
        IRREGULAR_DATES[: len(vals)],
        vals,
        lambda x: exponential_moving_average(x, float("nan")),
        rust_error="Beta",
        upstream_note="pandas_nan_beta_domain_bypass",
    )

payload = {
    "schema": "gsquant-ema-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/technicals.py:exponential_moving_average",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "ema-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream ema cases")
