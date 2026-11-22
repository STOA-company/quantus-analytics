"""Generate upstream backtest_basket / basket_series parity vectors only. Run with the locked upstream Python environment."""
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import warnings

# The caller supplies the fixed checkout; no network or Rust execution here.
sys.path.insert(0, sys.argv[1])
import numpy as np
import pandas as pd
from gs_quant.timeseries.backtesting import RebalFreq, backtest_basket, basket_series

ROOT = pathlib.Path(__file__).resolve().parent

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")

REBAL_MAP = {
    "daily": RebalFreq.DAILY,
    "weekly": RebalFreq.WEEKLY,
    "monthly": RebalFreq.MONTHLY,
}


def pack(v: float):
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return float(v)


def fmt_date(d) -> str:
    if isinstance(d, pd.Timestamp):
        return d.date().isoformat()
    if isinstance(d, dt.date):
        return d.isoformat()
    return str(d)


def make_series(dates, values):
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


def encode_series(dates, values):
    return {
        "dates": [d.isoformat() for d in dates],
        "values": [pack(float(v)) for v in values],
    }


def encode_param(p):
    if isinstance(p, tuple):
        dates, values = p
        return {
            "kind": "series",
            "dates": [d.isoformat() for d in dates],
            "values": [pack(float(v)) for v in values],
        }
    return {"kind": "fixed", "value": pack(float(p))}


def to_py_param(p):
    if isinstance(p, tuple):
        dates, values = p
        return make_series(dates, values)
    return float(p)


cases = []


def add_case(
    case_id: str,
    source_ref: str,
    operation: str,
    rebal_freq: str,
    series_specs: list,
    weights_spec=None,
    costs_spec=None,
    *,
    rust_error=None,
    documented_diff=None,
    expected_rust_error=None,
):
    py_series = [make_series(d, v) for d, v in series_specs]
    py_weights = None if weights_spec is None else [to_py_param(p) for p in weights_spec]
    py_costs = None if costs_spec is None else [to_py_param(p) for p in costs_spec]

    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": operation,
        "rebal_freq": rebal_freq,
        "series": [encode_series(d, v) for d, v in series_specs],
        "weights": None if weights_spec is None else [encode_param(p) for p in weights_spec],
        "costs": None if costs_spec is None else [encode_param(p) for p in costs_spec],
    }
    if documented_diff is not None:
        entry["documented_diff"] = documented_diff
        entry["expected_rust_error"] = expected_rust_error

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if operation == "backtest_basket":
                nav_s, aw_df = backtest_basket(
                    py_series, py_weights, py_costs, REBAL_MAP[rebal_freq]
                )
            elif operation == "basket_series":
                nav_s = basket_series(
                    py_series, py_weights, py_costs, REBAL_MAP[rebal_freq]
                )
                aw_df = None
            else:
                raise AssertionError(f"unknown operation: {operation}")
        except Exception as exc:
            if rust_error is None:
                raise
            entry["error"] = {
                "rust": rust_error,
                "upstream_type": type(exc).__name__,
            }
        else:
            if rust_error is not None:
                raise AssertionError(
                    f"{case_id}: expected upstream error for {rust_error}, got output"
                )
            dates = [fmt_date(d) for d in nav_s.index]
            if operation == "backtest_basket":
                entry["output"] = {
                    "dates": dates,
                    "nav": [pack(float(v)) for v in nav_s.values],
                    "actual_weights": [
                        [pack(float(v)) for v in aw_df.iloc[:, col].values]
                        for col in range(aw_df.shape[1])
                    ],
                }
            else:
                entry["output"] = {
                    "dates": dates,
                    "values": [pack(float(v)) for v in nav_s.values],
                }
    cases.append(entry)


# Shared deterministic synthetic dates and prices
D5 = [dt.date(2024, 1, d) for d in range(1, 6)]
P1_5 = [100.0, 110.0, 99.0, 108.9, 120.0]
P2_5 = [50.0, 50.0, 55.0, 52.25, 60.0]
P3_5 = [200.0, 220.0, 210.0, 231.0, 240.0]

# 20 business-like dates across Jan-Feb 2024
D20 = [
    dt.date(2024, 1, 2),
    dt.date(2024, 1, 3),
    dt.date(2024, 1, 4),
    dt.date(2024, 1, 5),
    dt.date(2024, 1, 8),
    dt.date(2024, 1, 9),
    dt.date(2024, 1, 10),
    dt.date(2024, 1, 11),
    dt.date(2024, 1, 12),
    dt.date(2024, 1, 16),
    dt.date(2024, 1, 17),
    dt.date(2024, 1, 18),
    dt.date(2024, 1, 19),
    dt.date(2024, 1, 22),
    dt.date(2024, 1, 23),
    dt.date(2024, 1, 24),
    dt.date(2024, 1, 25),
    dt.date(2024, 1, 26),
    dt.date(2024, 1, 29),
    dt.date(2024, 2, 5),
]
PA_20 = [
    100.0, 102.0, 101.0, 104.0, 107.0, 105.5, 108.0, 110.0, 109.0, 112.0,
    111.0, 115.0, 114.0, 118.0, 116.0, 119.5, 121.0, 120.0, 124.0, 128.0,
]
PB_20 = [
    80.0, 79.0, 81.5, 82.0, 80.0, 83.0, 84.5, 83.0, 85.0, 86.0,
    88.0, 87.0, 89.5, 88.0, 90.0, 91.0, 89.0, 92.5, 93.0, 95.0,
]
PC_20 = [
    250.0, 255.0, 252.0, 248.0, 260.0, 262.0, 258.0, 265.0, 270.0, 268.0,
    275.0, 272.0, 280.0, 285.0, 282.0, 288.0, 290.0, 287.0, 295.0, 300.0,
]

# Multi-month dates (Jan - Apr 2024)
D_MON = [
    dt.date(2024, 1, 10),
    dt.date(2024, 1, 22),
    dt.date(2024, 2, 5),
    dt.date(2024, 2, 12),
    dt.date(2024, 2, 26),
    dt.date(2024, 3, 11),
    dt.date(2024, 3, 25),
    dt.date(2024, 4, 10),
    dt.date(2024, 4, 22),
]
PA_MON = [100.0, 105.0, 103.0, 110.0, 114.0, 112.0, 120.0, 125.0, 130.0]
PB_MON = [50.0, 49.0, 52.0, 51.0, 54.0, 56.0, 55.0, 58.0, 60.0]
PC_MON = [200.0, 206.0, 210.0, 204.0, 212.0, 218.0, 222.0, 220.0, 232.0]

# 1. Upstream test_basket_series representative scenarios (8 cases)
add_case(
    "upstream-1-asset-daily",
    "test_backtesting.py::test_basket_series",
    "backtest_basket",
    "daily",
    [(D5, P1_5)],
)
add_case(
    "upstream-2-assets-equal-daily",
    "test_backtesting.py::test_basket_series",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
)
add_case(
    "upstream-3-assets-equal-daily",
    "test_backtesting.py::test_basket_series",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5), (D5, P3_5)],
)
add_case(
    "upstream-2-assets-w10-daily",
    "test_backtesting.py::test_basket_series",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[1.0, 0.0],
)
add_case(
    "upstream-2-assets-w01-daily",
    "test_backtesting.py::test_basket_series",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[0.0, 1.0],
)
add_case(
    "upstream-1-asset-monthly",
    "test_backtesting.py::test_basket_series",
    "backtest_basket",
    "monthly",
    [(D_MON, PA_MON)],
)
add_case(
    "upstream-2-assets-leveraged-1-1-weekly",
    "test_backtesting.py::test_basket_series",
    "backtest_basket",
    "weekly",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[1.0, 1.0],
)
add_case(
    "upstream-basket-series-2-assets",
    "test_backtesting.py::test_basket_series",
    "basket_series",
    "weekly",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[0.6, 0.4],
    costs_spec=[0.001, 0.002],
)

# 2. Rebalance schedules, exact max(cal) exclusion, gap snapping, Jan 31 leap/non-leap anchors (11 cases)
add_case(
    "sched-daily-2assets-no-cost",
    "synthetic-schedules",
    "backtest_basket",
    "daily",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[0.6, 0.4],
)
add_case(
    "sched-weekly-2assets-no-cost",
    "synthetic-schedules",
    "backtest_basket",
    "weekly",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[0.6, 0.4],
)
add_case(
    "sched-monthly-2assets-no-cost",
    "synthetic-schedules",
    "backtest_basket",
    "monthly",
    [(D_MON, PA_MON), (D_MON, PB_MON)],
    weights_spec=[0.6, 0.4],
)

D_EXACT7 = [dt.date(2024, 1, 1), dt.date(2024, 1, 4), dt.date(2024, 1, 8)]
add_case(
    "sched-weekly-exact-7d-max-excluded",
    "synthetic-schedules",
    "backtest_basket",
    "weekly",
    [(D_EXACT7, [100.0, 120.0, 150.0]), (D_EXACT7, [100.0, 100.0, 100.0])],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.1, 0.1],
)

D_EXACT14 = [
    dt.date(2024, 1, 1),
    dt.date(2024, 1, 4),
    dt.date(2024, 1, 8),
    dt.date(2024, 1, 11),
    dt.date(2024, 1, 15),
]
add_case(
    "sched-weekly-exact-14d-max-excluded",
    "synthetic-schedules",
    "backtest_basket",
    "weekly",
    [
        (D_EXACT14, [100.0, 115.0, 130.0, 140.0, 160.0]),
        (D_EXACT14, [100.0, 100.0, 100.0, 100.0, 100.0]),
    ],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.05, 0.05],
)

D_EXACT1M = [dt.date(2024, 1, 15), dt.date(2024, 2, 1), dt.date(2024, 2, 15)]
add_case(
    "sched-monthly-exact-1m-max-excluded",
    "synthetic-schedules",
    "backtest_basket",
    "monthly",
    [(D_EXACT1M, [100.0, 120.0, 140.0]), (D_EXACT1M, [100.0, 100.0, 100.0])],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.1, 0.1],
)

D_GAP_LAST = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 10)]
add_case(
    "sched-weekly-gap-snap-to-last",
    "synthetic-schedules",
    "backtest_basket",
    "weekly",
    [(D_GAP_LAST, [100.0, 120.0, 150.0]), (D_GAP_LAST, [100.0, 100.0, 100.0])],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.1, 0.1],
)

D_GAP_DEDUP = [
    dt.date(2024, 1, 1),
    dt.date(2024, 1, 2),
    dt.date(2024, 1, 25),
    dt.date(2024, 1, 26),
]
add_case(
    "sched-weekly-multi-gap-dedup",
    "synthetic-schedules",
    "backtest_basket",
    "weekly",
    [
        (D_GAP_DEDUP, [100.0, 120.0, 150.0, 165.0]),
        (D_GAP_DEDUP, [100.0, 100.0, 100.0, 100.0]),
    ],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.1, 0.1],
)

D_JAN31_LEAP = [
    dt.date(2024, 1, 31),
    dt.date(2024, 2, 28),
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 29),
    dt.date(2024, 3, 31),
    dt.date(2024, 4, 29),
    dt.date(2024, 4, 30),
    dt.date(2024, 5, 1),
]
add_case(
    "sched-monthly-jan31-leap-2024",
    "synthetic-schedules",
    "backtest_basket",
    "monthly",
    [
        (D_JAN31_LEAP, [100.0, 120.0, 150.0, 180.0, 200.0, 220.0, 250.0, 260.0]),
        (D_JAN31_LEAP, [100.0] * len(D_JAN31_LEAP)),
    ],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.02, 0.02],
)

D_JAN31_NONLEAP = [
    dt.date(2023, 1, 31),
    dt.date(2023, 2, 27),
    dt.date(2023, 2, 28),
    dt.date(2023, 3, 28),
    dt.date(2023, 3, 31),
    dt.date(2023, 4, 29),
    dt.date(2023, 4, 30),
    dt.date(2023, 5, 1),
]
add_case(
    "sched-monthly-jan31-nonleap-2023",
    "synthetic-schedules",
    "backtest_basket",
    "monthly",
    [
        (D_JAN31_NONLEAP, [100.0, 115.0, 140.0, 170.0, 190.0, 210.0, 230.0, 240.0]),
        (D_JAN31_NONLEAP, [100.0] * len(D_JAN31_NONLEAP)),
    ],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.02, 0.02],
)

D_MON_GAP = [
    dt.date(2024, 1, 10),
    dt.date(2024, 1, 15),
    dt.date(2024, 4, 15),
    dt.date(2024, 4, 16),
]
add_case(
    "sched-monthly-long-gap-dedup",
    "synthetic-schedules",
    "backtest_basket",
    "monthly",
    [
        (D_MON_GAP, [100.0, 110.0, 160.0, 168.0]),
        (D_MON_GAP, [100.0, 100.0, 100.0, 100.0]),
    ],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.05, 0.05],
)

# 3. Fixed costs, time-varying Series weights & costs, mixed Fixed + Series (8 cases)
add_case(
    "costs-fixed-daily-2assets",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "daily",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[0.6, 0.4],
    costs_spec=[0.001, 0.002],
)
add_case(
    "costs-fixed-weekly-3assets",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "weekly",
    [(D20, PA_20), (D20, PB_20), (D20, PC_20)],
    weights_spec=[0.5, 0.3, 0.2],
    costs_spec=[0.001, 0.0005, 0.002],
)
add_case(
    "costs-fixed-monthly-asymmetric",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "monthly",
    [(D_MON, PA_MON), (D_MON, PB_MON)],
    weights_spec=[0.7, 0.3],
    costs_spec=[0.0, 0.01],
)
add_case(
    "weights-series-daily",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[
        (D5, [0.5, 0.6, 0.7, 0.4, 0.5]),
        (D5, [0.5, 0.4, 0.3, 0.6, 0.5]),
    ],
    costs_spec=[0.002, 0.001],
)
add_case(
    "weights-series-weekly",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "weekly",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[
        (D20, [0.5 + 0.01 * i for i in range(20)]),
        (D20, [0.5 - 0.01 * i for i in range(20)]),
    ],
    costs_spec=[0.001, 0.001],
)
add_case(
    "costs-series-daily",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[0.6, 0.4],
    costs_spec=[
        (D5, [0.001, 0.002, 0.005, 0.001, 0.003]),
        (D5, [0.002, 0.001, 0.001, 0.004, 0.002]),
    ],
)
add_case(
    "params-mixed-fixed-and-series",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[
        (D5, [0.6, 0.8, 0.5, 0.7, 0.6]),
        0.4,
    ],
    costs_spec=[
        0.001,
        (D5, [0.001, 0.003, 0.002, 0.004, 0.001]),
    ],
)
add_case(
    "params-all-series-monthly",
    "synthetic-costs-and-series-params",
    "backtest_basket",
    "monthly",
    [(D_MON, PA_MON), (D_MON, PB_MON)],
    weights_spec=[
        (D_MON, [0.6, 0.55, 0.5, 0.65, 0.7, 0.4, 0.45, 0.5, 0.6]),
        (D_MON, [0.4, 0.45, 0.5, 0.35, 0.3, 0.6, 0.55, 0.5, 0.4]),
    ],
    costs_spec=[
        (D_MON, [0.001, 0.001, 0.002, 0.002, 0.003, 0.001, 0.002, 0.005, 0.001]),
        (D_MON, [0.002, 0.002, 0.001, 0.003, 0.001, 0.002, 0.001, 0.002, 0.003]),
    ],
)

# 4. Misaligned calendars & common date intersections (4 cases)
add_case(
    "align-prices-misaligned-daily",
    "synthetic-alignment",
    "backtest_basket",
    "daily",
    [
        ([dt.date(2024, 1, d) for d in (1, 2, 3, 4, 5, 8)], [95.0, 100.0, 105.0, 102.0, 108.0, 112.0]),
        ([dt.date(2024, 1, d) for d in (2, 3, 5, 6, 8, 9)], [50.0, 52.0, 54.0, 53.0, 57.0, 58.0]),
        ([dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)], [200.0, 210.0, 208.0, 215.0, 220.0]),
    ],
    weights_spec=[0.4, 0.35, 0.25],
    costs_spec=[0.001, 0.001, 0.001],
)
add_case(
    "align-prices-and-weight-series",
    "synthetic-alignment",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[
        ([dt.date(2024, 1, d) for d in (2, 3, 4)], [0.6, 0.7, 0.5]),
        0.4,
    ],
    costs_spec=[0.001, 0.002],
)
add_case(
    "align-prices-and-cost-series",
    "synthetic-alignment",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[0.5, 0.5],
    costs_spec=[
        0.001,
        ([dt.date(2024, 1, d) for d in (1, 3, 5)], [0.002, 0.004, 0.001]),
    ],
)
add_case(
    "align-all-params-weekly",
    "synthetic-alignment",
    "backtest_basket",
    "weekly",
    [
        (D20, PA_20),
        (D20[1:], PB_20[1:]),
    ],
    weights_spec=[
        (D20[:18], [0.55 + 0.01 * i for i in range(18)]),
        0.45,
    ],
    costs_spec=[
        0.001,
        (D20[2:19], [0.001 + 0.0005 * (i % 4) for i in range(17)]),
    ],
)

# 5. Single-asset, single-date, short / leveraged / all-zero / sum != 1 weights (9 cases)
add_case(
    "single-asset-with-costs-daily",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "daily",
    [(D5, P1_5)],
    weights_spec=[1.0],
    costs_spec=[0.01],
)
add_case(
    "single-asset-leveraged-with-costs-daily",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "daily",
    [(D5, P1_5)],
    weights_spec=[2.0],
    costs_spec=[0.01],
)
D_SINGLE = [dt.date(2024, 3, 1)]
add_case(
    "single-date-daily",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "daily",
    [(D_SINGLE, [123.45]), (D_SINGLE, [67.89])],
    weights_spec=[0.7, 0.3],
    costs_spec=[0.01, 0.02],
)
add_case(
    "single-date-weekly",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "weekly",
    [(D_SINGLE, [123.45]), (D_SINGLE, [67.89])],
    weights_spec=[0.7, 0.3],
    costs_spec=[0.01, 0.02],
)
add_case(
    "single-date-monthly",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "monthly",
    [(D_SINGLE, [123.45]), (D_SINGLE, [67.89])],
    weights_spec=[0.7, 0.3],
    costs_spec=[0.01, 0.02],
)
add_case(
    "short-long-150-neg50-weekly",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "weekly",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[1.5, -0.5],
    costs_spec=[0.001, 0.002],
)
add_case(
    "all-short-neg50-neg50-daily",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[-0.5, -0.5],
    costs_spec=[0.001, 0.001],
)
add_case(
    "all-zero-weights-daily",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[0.0, 0.0],
    costs_spec=[0.01, 0.01],
)
add_case(
    "sum-gt-one-3assets-monthly",
    "synthetic-weights-and-shapes",
    "backtest_basket",
    "monthly",
    [(D_MON, PA_MON), (D_MON, PB_MON), (D_MON, PC_MON)],
    weights_spec=[0.8, 0.6, 0.4],
    costs_spec=[0.001, 0.001, 0.002],
)

# 6. Zero NAV, negative NAV, NaN / zero prices IEEE-754 propagation (9 cases)
D3 = [dt.date(2024, 4, 1), dt.date(2024, 4, 2), dt.date(2024, 4, 3)]
add_case(
    "zero-nav-crash-daily",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "daily",
    [(D3, [100.0, 0.0, 10.0])],
    weights_spec=[1.0],
)
add_case(
    "negative-nav-leveraged-no-cost",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "daily",
    [(D3, [100.0, 40.0, 50.0])],
    weights_spec=[2.0],
)
add_case(
    "negative-nav-leveraged-with-cost-daily",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "daily",
    [(D3, [100.0, 40.0, 50.0])],
    weights_spec=[2.0],
    costs_spec=[0.01],
)
add_case(
    "nan-price-mid-daily",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "daily",
    [(D5, [100.0, 110.0, NAN, 108.0, 115.0]), (D5, P2_5)],
    weights_spec=[0.5, 0.5],
)
add_case(
    "nan-weight-series-daily",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[(D5, [0.5, 0.6, NAN, 0.5, 0.5]), 0.5],
    costs_spec=[0.001, 0.001],
)
add_case(
    "zero-price-day0-daily",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "daily",
    [(D3, [0.0, 10.0, 20.0]), (D3, [100.0, 105.0, 110.0])],
    weights_spec=[0.5, 0.5],
)
add_case(
    "zero-price-day0-weekly",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "weekly",
    [(D3, [0.0, 10.0, 20.0]), (D3, [100.0, 105.0, 110.0])],
    weights_spec=[0.5, 0.5],
)
add_case(
    "zero-price-day0-short-weekly",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "weekly",
    [(D3, [0.0, 10.0, 20.0]), (D3, [100.0, 105.0, 110.0])],
    weights_spec=[-0.5, 0.5],
)
D_ZMID = [
    dt.date(2024, 1, 1),
    dt.date(2024, 1, 2),
    dt.date(2024, 1, 3),
    dt.date(2024, 1, 8),
    dt.date(2024, 1, 9),
]
add_case(
    "zero-price-mid-nonrebal-weekly",
    "synthetic-nonfinite-and-nav-signs",
    "backtest_basket",
    "weekly",
    [
        (D_ZMID, [100.0, 0.0, 50.0, 80.0, 90.0]),
        (D_ZMID, [100.0, 100.0, 100.0, 100.0, 100.0]),
    ],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.001, 0.001],
)

# 7. basket_series wrapper cases (3 cases)
add_case(
    "basket-series-daily-default",
    "synthetic-basket-series",
    "basket_series",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
)
add_case(
    "basket-series-weekly-series-params",
    "synthetic-basket-series",
    "basket_series",
    "weekly",
    [(D20, PA_20), (D20, PB_20)],
    weights_spec=[(D20, [0.5 + 0.01 * i for i in range(20)]), 0.4],
    costs_spec=[0.001, 0.002],
)
add_case(
    "basket-series-monthly-jan31",
    "synthetic-basket-series",
    "basket_series",
    "monthly",
    [
        (D_JAN31_LEAP, [100.0, 120.0, 150.0, 180.0, 200.0, 220.0, 250.0, 260.0]),
        (D_JAN31_LEAP, [100.0] * len(D_JAN31_LEAP)),
    ],
    weights_spec=[0.5, 0.5],
    costs_spec=[0.02, 0.02],
)

# 8. Error cases (9 cases) + documented parity boundaries (2 cases)
add_case(
    "err-empty-series-default",
    "synthetic-errors",
    "backtest_basket",
    "daily",
    [],
    rust_error="Length",
)
add_case(
    "err-empty-series-with-weights",
    "synthetic-errors",
    "backtest_basket",
    "daily",
    [],
    weights_spec=[1.0],
    rust_error="Length",
)
add_case(
    "err-weights-too-short",
    "synthetic-errors",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[0.5],
    rust_error="Length",
)
add_case(
    "err-weights-too-long",
    "synthetic-errors",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[0.4, 0.4, 0.2],
    rust_error="Length",
)
add_case(
    "err-costs-too-short",
    "synthetic-errors",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    costs_spec=[0.001],
    rust_error="Length",
)
D_DISJ_A = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
D_DISJ_B = [dt.date(2024, 2, 1), dt.date(2024, 2, 2)]
add_case(
    "err-disjoint-prices-daily",
    "synthetic-errors",
    "backtest_basket",
    "daily",
    [(D_DISJ_A, [100.0, 101.0]), (D_DISJ_B, [50.0, 51.0])],
    rust_error="Length",
)
add_case(
    "err-disjoint-prices-weekly",
    "synthetic-errors",
    "backtest_basket",
    "weekly",
    [(D_DISJ_A, [100.0, 101.0]), (D_DISJ_B, [50.0, 51.0])],
    rust_error="Length",
)
add_case(
    "err-disjoint-weight-series",
    "synthetic-errors",
    "backtest_basket",
    "daily",
    [(D_DISJ_A, [100.0, 101.0])],
    weights_spec=[(D_DISJ_B, [1.0, 1.0])],
    rust_error="Length",
)
add_case(
    "err-empty-single-series",
    "synthetic-errors",
    "basket_series",
    "daily",
    [([], [])],
    rust_error="Length",
)
# Documented parity boundary: explicit empty weights/costs slice Some(&[]) returns Err(Error::Length)
# in Rust, whereas upstream Python's falsy `weights or [...]` / `costs or [...]` falls back to defaults.
add_case(
    "doc-empty-weights-slice",
    "documented-parity-boundaries",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[],
    documented_diff="empty_slice_vs_python_falsy_default",
    expected_rust_error="Length",
)
add_case(
    "doc-empty-costs-slice",
    "documented-parity-boundaries",
    "backtest_basket",
    "daily",
    [(D5, P1_5), (D5, P2_5)],
    weights_spec=[0.6, 0.4],
    costs_spec=[],
    documented_diff="empty_slice_vs_python_falsy_default",
    expected_rust_error="Length",
)

payload = {
    "schema": "gsquant-basket-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/timeseries/backtesting.py:backtest_basket",
        "gs_quant/timeseries/backtesting.py:basket_series",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "basket-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream basket cases")
