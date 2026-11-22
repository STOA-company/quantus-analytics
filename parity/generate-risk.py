"""Generate upstream risk aggregation, subtraction, sorting & point_sort_order parity vectors. Run with the locked upstream Python environment."""
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
from gs_quant.base import RiskKey
from gs_quant.common import RiskRequestParameters
from gs_quant.datetime.point import point_sort_order
from gs_quant.risk.core import (
    DataFrameWithInfo,
    aggregate_results,
    aggregate_risk,
    sort_risk,
    subtract_risk,
)

ROOT = pathlib.Path(__file__).resolve().parent
REF_DATE = dt.date(2024, 1, 15)
REF_DATE_ISO = REF_DATE.isoformat()

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")


def pack(v: float):
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return float(v)


def fmt_date(d):
    if d is None:
        return None
    if isinstance(d, pd.Timestamp):
        return d.date().isoformat()
    if isinstance(d, dt.date):
        return d.isoformat()
    return str(d)


def make_row(mkt_type, mkt_asset, mkt_class, mkt_point, value, date=None, time=None):
    return {
        "date": fmt_date(date),
        "time": time,
        "mkt_type": mkt_type,
        "mkt_asset": mkt_asset,
        "mkt_class": mkt_class,
        "mkt_point": mkt_point,
        "value": pack(float(value)),
    }


def make_table(rows, unit=None):
    return {"unit": unit, "rows": rows}


def unpack_float(v) -> float:
    if v == "NaN":
        return float("nan")
    if v == "Infinity":
        return float("inf")
    if v == "-Infinity":
        return float("-inf")
    return float(v)


def table_to_df(table_spec):
    rows = table_spec["rows"]
    has_date = any(r["date"] is not None for r in rows)
    has_time = any(r["time"] is not None for r in rows)
    cols = []
    if has_date:
        cols.append("date")
    if has_time:
        cols.append("time")
    cols.extend(["mkt_type", "mkt_asset", "mkt_class", "mkt_point", "value"])
    records = []
    for r in rows:
        rec = {}
        if has_date:
            rec["date"] = dt.date.fromisoformat(r["date"])
        if has_time:
            rec["time"] = r["time"]
        rec["mkt_type"] = r["mkt_type"]
        rec["mkt_asset"] = r["mkt_asset"]
        rec["mkt_class"] = r["mkt_class"]
        rec["mkt_point"] = r["mkt_point"]
        rec["value"] = unpack_float(r["value"])
        records.append(rec)
    df = pd.DataFrame(records, columns=cols)
    unit_dict = {table_spec["unit"]: 1} if table_spec["unit"] else None
    rk = RiskKey(None, None, None, RiskRequestParameters(), None, "IRDelta")
    return DataFrameWithInfo(df, risk_key=rk, unit=unit_dict)


def df_to_rows(df: pd.DataFrame):
    if df.index.name == "date":
        df = df.reset_index()
    out = []
    has_date = "date" in df.columns
    has_time = "time" in df.columns
    for _, row in df.iterrows():
        out.append(
            {
                "date": fmt_date(row["date"]) if has_date else None,
                "time": str(row["time"]) if has_time else None,
                "mkt_type": str(row["mkt_type"]),
                "mkt_asset": str(row["mkt_asset"]),
                "mkt_class": str(row["mkt_class"]),
                "mkt_point": str(row["mkt_point"]),
                "value": pack(float(row["value"])),
            }
        )
    return out


def unit_dict_to_opt(u):
    if not u:
        return None
    keys = list(u.keys())
    return keys[0] if keys else None


cases = []


def add_point_case(case_id: str, source_ref: str, point: str, ref_date: dt.date = REF_DATE):
    out = point_sort_order(point, ref_date)
    cases.append(
        {
            "id": case_id,
            "source_ref": source_ref,
            "operation": "point_sort_order",
            "ref_date": ref_date.isoformat(),
            "point": point,
            "output": None if out is None else pack(float(out)),
        }
    )


def add_sort_case(case_id: str, source_ref: str, table_spec: dict):
    df = table_to_df(table_spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sort_risk(df)
    cases.append(
        {
            "id": case_id,
            "source_ref": source_ref,
            "operation": "sort_risk",
            "ref_date": REF_DATE_ISO,
            "table": table_spec,
            "output": {
                "unit": table_spec["unit"],
                "rows": df_to_rows(res),
            },
        }
    )


def add_aggregate_case(
    case_id: str,
    source_ref: str,
    table_specs: list,
    threshold=None,
    *,
    use_aggregate_results: bool = False,
    rust_error=None,
):
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "aggregate_risk",
        "ref_date": REF_DATE_ISO,
        "threshold": None if threshold is None else pack(float(threshold)),
        "tables": table_specs,
    }
    dfs = [table_to_df(t) for t in table_specs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if use_aggregate_results:
                res = aggregate_results(dfs)
                res_unit = unit_dict_to_opt(res.unit)
                res_rows = df_to_rows(res)
            else:
                res = aggregate_risk(dfs, threshold=threshold)
                non_empty_units = [t["unit"] for t in table_specs if t["unit"]]
                res_unit = non_empty_units[0] if non_empty_units else None
                res_rows = df_to_rows(res)
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
            entry["output"] = {"unit": res_unit, "rows": res_rows}
    cases.append(entry)


def add_subtract_case(
    case_id: str,
    source_ref: str,
    left_spec: dict,
    right_spec: dict,
    *,
    rust_error=None,
    upstream_note=None,
):
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "subtract_risk",
        "ref_date": REF_DATE_ISO,
        "left": left_spec,
        "right": right_spec,
    }
    if upstream_note is not None:
        entry["upstream_note"] = upstream_note

    left_df = table_to_df(left_spec)
    right_df = table_to_df(right_spec)

    # Verify upstream subtract_risk raises AssertionError on standard DataFrames (core.py:642-643 bug)
    if upstream_note == "upstream_subtract_risk_columns_names_bug":
        try:
            subtract_risk(left_df, right_df)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"{case_id}: expected upstream subtract_risk AssertionError"
            )

    if rust_error is not None:
        if rust_error == "UnitMismatch":
            # Verify upstream aggregate_results raises ValueError on mismatched units
            try:
                aggregate_results((left_df, right_df))
            except ValueError as exc:
                entry["error"] = {
                    "rust": rust_error,
                    "upstream_type": type(exc).__name__,
                }
            else:
                raise AssertionError(f"{case_id}: expected ValueError for UnitMismatch")
        else:
            entry["error"] = {
                "rust": rust_error,
                "upstream_type": "AssertionError",
            }
        cases.append(entry)
        return

    # Compute exact upstream expected values via aggregate_risk((left, right_negated))
    right_neg_raw = right_df.raw_value.copy()
    right_neg_raw["value"] = -right_neg_raw["value"]
    right_neg_df = DataFrameWithInfo(
        right_neg_raw, risk_key=right_df.risk_key, unit=right_df.unit
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = aggregate_risk((left_df, right_neg_df))
    non_empty_units = [u for u in (left_spec["unit"], right_spec["unit"]) if u]
    res_unit = non_empty_units[0] if non_empty_units else None
    entry["output"] = {"unit": res_unit, "rows": df_to_rows(res)}
    cases.append(entry)


# =============================================================================
# 1. point_sort_order cases (52 cases covering every pattern branch in point.py)
# =============================================================================
POINT_CASES = [
    ("pso-empty", ""),
    ("pso-const-on", "O/N"),
    ("pso-const-on-lower", "o/n"),
    ("pso-const-tn", "T/N"),
    ("pso-const-ois-fix", "OIS FIX"),
    ("pso-const-cash-stub", "Cash Stub"),
    ("pso-const-cashstub", "CashStub"),
    ("pso-const-default", "DEFAULT"),
    ("pso-const-in", "In"),
    ("pso-const-out", "Out"),
    ("pso-compound-1y-5y", "1y;5y"),
    ("pso-compound-1y-2y-3y", "1y;2y;3y"),
    ("pso-compound-6m-2y-10y", "6m;2y;10y"),
    ("pso-compound-on-5y", "O/N;5y"),
    ("pso-compound-spaced", " 3m ; 1y "),
    ("pso-infl-caplet", "USD Caplet 1y"),
    ("pso-infl-zccap", "ZCCap 5y"),
    ("pso-infl-swaption", "Swaption 10y"),
    ("pso-infl-zcswo", "ZCSwo Caplet"),
    ("pso-copula-rho", "USD Rho"),
    ("pso-copula-ko", "EUR KO"),
    ("pso-copula-rr", "1y RR"),
    ("pso-copula-bf", "BF 2y"),
    ("pso-copula-alpha", "Alpha"),
    ("pso-copula-k0l", "K0=L"),
    ("pso-seasonal-front", "Front"),
    ("pso-seasonal-back", "Back"),
    ("pso-seasonal-back-front", "Back Front"),
    ("pso-mmm-jan", "Jan"),
    ("pso-mmm-dec", "Dec"),
    ("pso-euro-jan25", "JAN25"),
    ("pso-euro-dec68", "Dec68"),
    ("pso-euro-dec69", "Dec69"),
    ("pso-rdate-1d", "1d"),
    ("pso-rdate-10b", "10B"),
    ("pso-rdate-neg-2w", "-2w"),
    ("pso-rdate-2f", "2F"),
    ("pso-rdate-1y5y", "1y5y"),
    ("pso-cashfx-1y5y-xc", "1y5y XC"),
    ("pso-pricerbf-1y5y10y", "1y5y10y"),
    ("pso-frax-3x6", "3x6"),
    ("pso-spike-qe1", "QE1-2025"),
    ("pso-spike-qe2", "QE2-2025"),
    ("pso-spike-qe3", "QE3-2025"),
    ("pso-spike-qe4", "QE4-2025"),
    ("pso-mmmyyyy-jun2025", "Jun2025"),
    ("pso-ddmmmyyyy-15mar2026", "15Mar2026"),
    ("pso-ddmmmyyyy-leap-29feb2024", "29Feb2024"),
    ("pso-ddmmmyyyy-1jan2024", "1Jan2024"),
    ("pso-ddmmmyyyy-leading-zero-miss", "05Jan2024"),
    ("pso-number-42", "42"),
    ("pso-floating-year-05y", "0.5y"),
    ("pso-floating-year-15Y", "1.5Y"),
    ("pso-floating-year-dot5y", ".5y"),
    ("pso-pricer-coord-no", "No 5"),
    ("pso-pricer-coord-pricer", "Pricer 12"),
    ("pso-bond-spread-none", "5S25/5Q25"),
    ("pso-lyy-h25", "H25"),
    ("pso-lyy-m26", "M26"),
    ("pso-lyy-z68", "Z68"),
    ("pso-lyy-z69", "Z69"),
    ("pso-datepair-slash", "20240101/20250615"),
    ("pso-datepair-space", "20240101 20250615"),
    ("pso-mmmyy-jan-25", "JAN 25"),
    ("pso-fxvol-spread-addon", "Spread Addon"),
    ("pso-fxvol-hl", "HL"),
    ("pso-bond-coord", "1 2.5 15/03/2026"),
    ("pso-bond-fut-usth4", "USTH4"),
    ("pso-ff-fut-ffz5", "FFZ5"),
    ("pso-ff-fut-ffh4", "FFH4"),
    ("pso-repogc-on", "ON GC"),
    ("pso-repogc-tn", "TN GC"),
    ("pso-repogc-sn", "SN GC"),
    ("pso-repogc-3m", "3 Month GC"),
    ("pso-repogc-2w", "2 Week GC"),
    ("pso-repogc-1y", "1 Year GC"),
    ("pso-repogc-5d", "5 Day GC"),
    ("pso-repogc-bare-none", "3 GC"),
    ("pso-relative-10d", "10 day"),
    ("pso-relative-3w", "3 week"),
    ("pso-relative-6m", "6 month"),
    ("pso-relative-2y-upper", "2 YEAR"),
    ("pso-ddmmmyy-05feb25", "05Feb25"),
    ("pso-ddmmmyy-leap-29feb24", "29Feb24"),
    ("pso-fallback-unknown", "unknown_point"),
]

for cid, pt in POINT_CASES:
    add_point_case(cid, "gs_quant/datetime/point.py:point_sort_order", pt, REF_DATE)


# =============================================================================
# 2. sort_risk cases (6 cases, relative tenors only -> date.today()-independent)
# =============================================================================
add_sort_case(
    "sort-basic-tenors",
    "gs_quant/risk/core.py:sort_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "10y", 100.0),
            make_row("IR", "USD", "SWAP", "3m", 25.0),
            make_row("IR", "USD", "SWAP", "O/N", 5.0),
            make_row("IR", "USD", "SWAP", "1y", 50.0),
            make_row("IR", "USD", "SWAP", "5y", 75.0),
        ],
        unit="USD",
    ),
)

add_sort_case(
    "sort-tie-stability-30d-1m",
    "gs_quant/risk/core.py:sort_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "5y", 10.0),
            make_row("IR", "USD", "SWAP", "30d", 5.0),
            make_row("IR", "USD", "SWAP", "1m", 7.0),
            make_row("IR", "USD", "SWAP", "1y", 20.0),
        ],
        unit="USD",
    ),
)

add_sort_case(
    "sort-none-point-as-zero",
    "gs_quant/risk/core.py:sort_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "1d", 3.0),
            make_row("IR", "USD", "SWAP", "USD Rho", 1.0),
            make_row("IR", "USD", "SWAP", "O/N", 2.0),
            make_row("IR", "USD", "SWAP", "-1w", 4.0),
        ],
        unit="USD",
    ),
)

add_sort_case(
    "sort-multi-column-dims",
    "gs_quant/risk/core.py:sort_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "1y", 1.0),
            make_row("FX", "USD", "FWD", "5y", 2.0),
            make_row("IR", "EUR", "SWAP", "5y", 3.0),
            make_row("IR", "USD", "CASH", "5y", 4.0),
            make_row("IR", "USD", "SWAP", "3m", 5.0),
        ],
        unit="USD",
    ),
)

add_sort_case(
    "sort-dated-and-timed",
    "gs_quant/risk/core.py:sort_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "5y", 10.0, date=dt.date(2024, 1, 2), time="09:00"),
            make_row("IR", "USD", "SWAP", "5y", 20.0, date=dt.date(2024, 1, 1), time="14:00"),
            make_row("IR", "USD", "SWAP", "1y", 30.0, date=dt.date(2024, 1, 1), time="09:00"),
            make_row("IR", "USD", "SWAP", "5y", 40.0, date=dt.date(2024, 1, 1), time="09:00"),
        ],
        unit="USD",
    ),
)

add_sort_case(
    "sort-compound-semicolon-tenors",
    "gs_quant/risk/core.py:sort_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAPTION", "1y;10y", 12.0),
            make_row("IR", "USD", "SWAPTION", "1y;2y", 8.0),
            make_row("IR", "USD", "SWAPTION", "6m;5y", 5.0),
            make_row("IR", "USD", "SWAPTION", "1y;5y", 10.0),
        ],
        unit="USD",
    ),
)


# =============================================================================
# 3. aggregate_risk & aggregate_results cases (20 cases)
# =============================================================================
T_A = make_table(
    [
        make_row("IR", "USD", "SWAP", "5y", 10.0),
        make_row("IR", "USD", "SWAP", "1y", 20.0),
        make_row("IR", "USD", "SWAP", "30d", 5.0),
        make_row("IR", "USD", "SWAP", "1m", 7.0),
    ],
    unit="USD",
)
T_B = make_table(
    [
        make_row("IR", "USD", "SWAP", "5y", -10.0),
        make_row("IR", "USD", "SWAP", "1y", NAN),
        make_row("IR", "USD", "SWAP", "2y", -25.0),
        make_row("IR", "USD", "SWAP", "10y", 10.0),
    ],
    unit="USD",
)
T_NONE_UNIT = make_table(
    [
        make_row("IR", "USD", "SWAP", "1y", 5.0),
        make_row("IR", "USD", "SWAP", "3m", 15.0),
    ],
    unit=None,
)
T_EUR = make_table(
    [
        make_row("IR", "EUR", "SWAP", "5y", 3.0),
    ],
    unit="EUR",
)

add_aggregate_case(
    "agg-two-tables-fillna-cancel-tiebreak",
    "gs_quant/risk/core.py:aggregate_risk",
    [T_A, T_B],
)
add_aggregate_case(
    "agg-threshold-zero-drops-cancel",
    "gs_quant/risk/core.py:aggregate_risk",
    [T_A, T_B],
    threshold=0.0,
)
add_aggregate_case(
    "agg-threshold-10-strict-boundary",
    "gs_quant/risk/core.py:aggregate_risk",
    [T_A, T_B],
    threshold=10.0,
)
add_aggregate_case(
    "agg-threshold-25-strict-boundary",
    "gs_quant/risk/core.py:aggregate_risk",
    [T_A, T_B],
    threshold=25.0,
)
add_aggregate_case(
    "agg-threshold-neg-1-keeps-zero",
    "gs_quant/risk/core.py:aggregate_risk",
    [T_A, T_B],
    threshold=-1.0,
)
add_aggregate_case(
    "agg-threshold-nan-drops-all",
    "gs_quant/risk/core.py:aggregate_risk",
    [T_A, T_B],
    threshold=NAN,
)
add_aggregate_case(
    "agg-threshold-pos-inf-drops-all",
    "gs_quant/risk/core.py:aggregate_risk",
    [T_A, T_B],
    threshold=INF,
)
add_aggregate_case(
    "agg-duplicate-buckets-single-table",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", 10.0),
                make_row("IR", "USD", "SWAP", "5y", 12.5),
                make_row("IR", "USD", "SWAP", "5y", -2.5),
                make_row("IR", "USD", "SWAP", "1y", NAN),
                make_row("IR", "USD", "SWAP", "1y", 4.0),
            ],
            unit="USD",
        )
    ],
)
add_aggregate_case(
    "agg-all-nan-bucket-becomes-zero",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "1y", NAN),
                make_row("IR", "USD", "SWAP", "5y", 3.0),
            ],
            unit="USD",
        ),
        make_table(
            [
                make_row("IR", "USD", "SWAP", "1y", NAN),
            ],
            unit="USD",
        ),
    ],
)
add_aggregate_case(
    "agg-inf-propagation",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "1y", INF),
                make_row("IR", "USD", "SWAP", "2y", NINF),
                make_row("IR", "USD", "SWAP", "5y", INF),
            ],
            unit="USD",
        ),
        make_table(
            [
                make_row("IR", "USD", "SWAP", "1y", 5.0),
                make_row("IR", "USD", "SWAP", "2y", -3.0),
                make_row("IR", "USD", "SWAP", "5y", NINF),
            ],
            unit="USD",
        ),
    ],
)
add_aggregate_case(
    "agg-inf-threshold-zero-drops-nan",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "1y", INF),
                make_row("IR", "USD", "SWAP", "5y", INF),
            ],
            unit="USD",
        ),
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", NINF),
            ],
            unit="USD",
        ),
    ],
    threshold=0.0,
)
add_aggregate_case(
    "agg-three-tables-multi-asset-class",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", 10.0),
                make_row("FX", "EURUSD", "FWD", "3m", -4.0),
            ],
            unit="USD",
        ),
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", 2.5),
                make_row("IR", "USD", "OIS", "1y", 8.0),
            ],
            unit="USD",
        ),
        make_table(
            [
                make_row("FX", "EURUSD", "FWD", "3m", 6.5),
                make_row("IR", "EUR", "SWAP", "2y", -1.25),
            ],
            unit="USD",
        ),
    ],
)
add_aggregate_case(
    "agg-dated-tables",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", 10.0, date=dt.date(2024, 1, 2)),
                make_row("IR", "USD", "SWAP", "1y", 20.0, date=dt.date(2024, 1, 1)),
            ],
            unit="USD",
        ),
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", 5.0, date=dt.date(2024, 1, 2)),
                make_row("IR", "USD", "SWAP", "5y", 7.0, date=dt.date(2024, 1, 1)),
            ],
            unit="USD",
        ),
    ],
)
add_aggregate_case(
    "agg-dated-and-timed-tables",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", 10.0, date=dt.date(2024, 1, 1), time="14:00"),
                make_row("IR", "USD", "SWAP", "5y", 20.0, date=dt.date(2024, 1, 1), time="09:00"),
            ],
            unit="USD",
        ),
        make_table(
            [
                make_row("IR", "USD", "SWAP", "5y", -5.0, date=dt.date(2024, 1, 1), time="09:00"),
            ],
            unit="USD",
        ),
    ],
)
add_aggregate_case(
    "agg-groupby-lex-tiebreak-30d-1m",
    "gs_quant/risk/core.py:aggregate_risk",
    [
        make_table(
            [
                make_row("IR", "USD", "SWAP", "30d", 5.0),
                make_row("IR", "USD", "SWAP", "1m", 7.0),
            ],
            unit="USD",
        )
    ],
)
add_aggregate_case(
    "agg-unit-inherit-none-and-usd",
    "gs_quant/risk/core.py:aggregate_results",
    [T_NONE_UNIT, T_A],
    use_aggregate_results=True,
)
add_aggregate_case(
    "agg-unit-all-none",
    "gs_quant/risk/core.py:aggregate_results",
    [
        T_NONE_UNIT,
        make_table([make_row("IR", "USD", "SWAP", "1y", 2.0)], unit=None),
    ],
    use_aggregate_results=True,
)
add_aggregate_case(
    "agg-err-unit-mismatch-usd-eur",
    "gs_quant/risk/core.py:aggregate_results",
    [T_A, T_EUR],
    use_aggregate_results=True,
    rust_error="UnitMismatch",
)
add_aggregate_case(
    "agg-err-empty-tables",
    "gs_quant/risk/core.py:aggregate_risk",
    [],
    rust_error="Length",
)
add_aggregate_case(
    "agg-single-empty-rows-table",
    "gs_quant/risk/core.py:aggregate_risk",
    [make_table([], unit="USD")],
)


# =============================================================================
# 4. subtract_risk cases (6 cases verifying upstream bug + intended semantics)
# =============================================================================
add_subtract_case(
    "sub-cancellation-and-asymmetric",
    "gs_quant/risk/core.py:subtract_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "5y", 10.0),
            make_row("IR", "USD", "SWAP", "1y", 25.0),
            make_row("IR", "USD", "SWAP", "3m", 4.0),
        ],
        unit="USD",
    ),
    make_table(
        [
            make_row("IR", "USD", "SWAP", "5y", 10.0),
            make_row("IR", "USD", "SWAP", "2y", 6.0),
            make_row("IR", "USD", "SWAP", "1y", 5.0),
        ],
        unit="USD",
    ),
    upstream_note="upstream_subtract_risk_columns_names_bug",
)
add_subtract_case(
    "sub-with-nan-fillna",
    "gs_quant/risk/core.py:subtract_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "1y", 15.0),
            make_row("IR", "USD", "SWAP", "5y", NAN),
        ],
        unit="USD",
    ),
    make_table(
        [
            make_row("IR", "USD", "SWAP", "1y", NAN),
            make_row("IR", "USD", "SWAP", "5y", 3.5),
        ],
        unit=None,
    ),
    upstream_note="upstream_subtract_risk_columns_names_bug",
)
add_subtract_case(
    "sub-dated-tables",
    "gs_quant/risk/core.py:subtract_risk",
    make_table(
        [
            make_row("IR", "USD", "SWAP", "5y", 30.0, date=dt.date(2024, 1, 2)),
            make_row("IR", "USD", "SWAP", "1y", 12.0, date=dt.date(2024, 1, 1)),
        ],
        unit="USD",
    ),
    make_table(
        [
            make_row("IR", "USD", "SWAP", "5y", 18.0, date=dt.date(2024, 1, 2)),
            make_row("IR", "USD", "SWAP", "1y", 12.0, date=dt.date(2024, 1, 1)),
        ],
        unit="USD",
    ),
    upstream_note="upstream_subtract_risk_columns_names_bug",
)
add_subtract_case(
    "sub-self-exact-zero",
    "gs_quant/risk/core.py:subtract_risk",
    T_A,
    T_A,
    upstream_note="upstream_subtract_risk_columns_names_bug",
)
add_subtract_case(
    "sub-err-unit-mismatch",
    "gs_quant/risk/core.py:subtract_risk",
    T_A,
    T_EUR,
    rust_error="UnitMismatch",
    upstream_note="upstream_subtract_risk_columns_names_bug",
)
add_subtract_case(
    "sub-err-date-schema-mismatch",
    "gs_quant/risk/core.py:subtract_risk",
    T_A,
    make_table(
        [make_row("IR", "USD", "SWAP", "5y", 1.0, date=dt.date(2024, 1, 15))],
        unit="USD",
    ),
    rust_error="Length",
    upstream_note="upstream_subtract_risk_columns_names_bug",
)

payload = {
    "schema": "gsquant-risk-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "numpy": __import__("numpy").__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/datetime/point.py:point_sort_order",
        "gs_quant/risk/core.py:sort_risk",
        "gs_quant/risk/core.py:aggregate_risk",
        "gs_quant/risk/core.py:aggregate_results",
        "gs_quant/risk/core.py:subtract_risk",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "risk-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream risk cases")
