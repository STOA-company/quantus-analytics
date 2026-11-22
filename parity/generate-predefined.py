"""Generate upstream PredefinedAssetEngine parity vectors only. Run with the locked upstream Python environment."""
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

import gs_quant.backtests.predefined_asset_engine as pae_mod
from gs_quant.backtests.backtest_objects import PredefinedAssetBacktest
from gs_quant.backtests.core import TimeWindow, ValuationFixingType
from gs_quant.backtests.data_sources import DataManager
from gs_quant.backtests.order import (
    OrderAtMarket,
    OrderCost,
    OrderMarketOnClose,
    OrderTWAP,
)
from gs_quant.backtests.predefined_asset_engine import PredefinedAssetEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.triggers import OrdersGeneratorTrigger
from gs_quant.data.core import DataFrequency
from gs_quant.instrument import Cash, IRBondFuture

ROOT = pathlib.Path(__file__).resolve().parent
NAN = float("nan")


def pack(v: float):
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return float(v)


def fmt_ts(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def fmt_date(d: dt.date) -> str:
    return d.isoformat()


def inst_label(inst, cash_label: str) -> str:
    if isinstance(inst, Cash):
        return cash_label
    return inst.name


def make_offline_calendar_patches(holidays: list):
    """Build numpy.busdaycalendar-backed offline replacements for gs_quant calendar helpers."""
    hol_arr = np.array([np.datetime64(h.isoformat(), "D") for h in holidays], dtype="datetime64[D]")
    cal = np.busdaycalendar(weekmask="1111100", holidays=hol_arr)

    def _to_date(np_d) -> dt.date:
        return dt.date.fromisoformat(str(np_d))

    def _is_bday(dates, calendars=None):
        if isinstance(dates, (list, tuple)):
            return [bool(np.is_busday(np.datetime64(d.isoformat(), "D"), busdaycal=cal)) for d in dates]
        return bool(np.is_busday(np.datetime64(dates.isoformat(), "D"), busdaycal=cal))

    def _bday_offset(dates, offsets, roll="raise", calendars=None):
        np_roll = {
            "raise": "raise",
            "forward": "forward",
            "following": "forward",
            "backward": "backward",
            "preceding": "backward",
            "modifiedfollowing": "modifiedfollowing",
            "modifiedpreceding": "modifiedpreceding",
        }[roll.lower()]
        res = np.busday_offset(
            np.datetime64(dates.isoformat(), "D"),
            offsets,
            roll=np_roll,
            busdaycal=cal,
        )
        return _to_date(res)

    def _prev_bday(date, calendars=None):
        return _bday_offset(date, -1, roll="forward", calendars=calendars)

    return cal, _is_bday, _bday_offset, _prev_bday


class ListOrdersTrigger(OrdersGeneratorTrigger):
    """Offline trigger emitting explicit pre-scheduled OrderBase instances at their generation_time."""

    def __init__(self, order_specs: list, instruments: dict, active_days: set, cash_label: str):
        super().__init__()
        self._order_specs = order_specs
        self._instruments = instruments
        self._active_days = active_days
        self._cash_label = cash_label

    def get_trigger_times(self) -> list:
        times = []
        for spec in self._order_specs:
            gen_ts = spec["generation_time"]
            if gen_ts.date() in self._active_days:
                times.append(gen_ts.time())
        return list(dict.fromkeys(times))

    def generate_orders(self, state: dt.datetime, backtest: PredefinedAssetBacktest = None) -> list:
        out = []
        for spec in self._order_specs:
            if spec["generation_time"] != state:
                continue
            kind = spec["kind"]
            qty = float(spec["quantity"])
            src = spec["source"]
            if kind == "MOC":
                inst = self._instruments[spec["instrument"]]
                out.append(
                    OrderMarketOnClose(
                        instrument=inst,
                        quantity=qty,
                        generation_time=state,
                        execution_date=spec["execution_date"],
                        source=src,
                    )
                )
            elif kind == "TWAP":
                inst = self._instruments[spec["instrument"]]
                out.append(
                    OrderTWAP(
                        instrument=inst,
                        quantity=qty,
                        generation_time=state,
                        window=TimeWindow(start=spec["start"], end=spec["end"]),
                        source=src,
                    )
                )
            elif kind == "AT_MARKET":
                inst = self._instruments[spec["instrument"]]
                out.append(
                    OrderAtMarket(
                        instrument=inst,
                        quantity=qty,
                        generation_time=state,
                        execution_datetime=spec["execution_time"],
                        source=src,
                    )
                )
            elif kind == "COST":
                out.append(
                    OrderCost(
                        currency=self._cash_label,
                        quantity=qty,
                        source=src,
                        execution_time=spec["execution_time"],
                    )
                )
            else:
                raise AssertionError(f"unknown order kind: {kind}")
        return out


def encode_order_spec(spec: dict) -> dict:
    base = {
        "kind": spec["kind"],
        "instrument": spec["instrument"],
        "quantity": pack(float(spec["quantity"])),
        "generation_time": fmt_ts(spec["generation_time"]),
        "source": spec["source"],
    }
    if spec["kind"] == "MOC":
        base["execution_date"] = fmt_date(spec["execution_date"])
    elif spec["kind"] == "TWAP":
        base["start"] = fmt_ts(spec["start"])
        base["end"] = fmt_ts(spec["end"])
    elif spec["kind"] in ("AT_MARKET", "COST"):
        base["execution_time"] = fmt_ts(spec["execution_time"])
    return base


cases = []


def add_case(
    case_id: str,
    source_ref: str,
    start: dt.date,
    end: dt.date,
    *,
    holidays=None,
    initial_value: float = 100.0,
    eod_valuation_time=(23, 0, 0),
    cash_label: str = "USD",
    daily_prices=None,
    intraday_prices=None,
    orders=None,
    rust_error=None,
):
    holidays = list(holidays or [])
    daily_prices = dict(daily_prices or {})
    intraday_prices = dict(intraday_prices or {})
    orders = list(orders or [])

    cal, fn_is_bday, fn_bday_offset, fn_prev_bday = make_offline_calendar_patches(holidays)

    # Collect all instrument names referenced anywhere
    inst_names = set(daily_prices.keys()) | set(intraday_prices.keys())
    for o in orders:
        if o["kind"] != "COST":
            inst_names.add(o["instrument"])
    instruments = {name: IRBondFuture(currency="EUR", name=name) for name in sorted(inst_names)}

    data_mgr = DataManager()
    for name in sorted(daily_prices.keys()):
        d_list, v_list = daily_prices[name]
        s = pd.Series(v_list, index=list(d_list), dtype=float)
        data_mgr.add_data_source(s, DataFrequency.DAILY, instruments[name], ValuationFixingType.PRICE)
    for name in sorted(intraday_prices.keys()):
        ticks = intraday_prices[name]
        s = pd.Series([v for _, v in ticks], index=[t for t, _ in ticks], dtype=float)
        data_mgr.add_data_source(s, DataFrequency.REAL_TIME, instruments[name], ValuationFixingType.PRICE)

    # Compute active days matching _adjust_date + (adjusted_start, adjusted_end]
    adj_start = fn_bday_offset(start, 0, roll="backward")
    adj_end = fn_bday_offset(end, 0, roll="backward")
    active_days = set()
    cur = adj_start + dt.timedelta(days=1)
    while cur <= adj_end:
        if fn_is_bday(cur):
            active_days.add(cur)
        cur += dt.timedelta(days=1)

    trigger = ListOrdersTrigger(orders, instruments, active_days, cash_label)
    strategy = Strategy(initial_portfolio=None, triggers=[trigger])
    engine = PredefinedAssetEngine(
        data_mgr=data_mgr,
        calendars="OFFLINE_CUSTOM",
        tz=dt.timezone.utc,
    )

    recorded_fills = []
    orig_is_bday = pae_mod.is_business_day
    orig_bday_offset = pae_mod.business_day_offset
    orig_prev_bday = pae_mod.prev_business_date
    orig_update_fill = PredefinedAssetBacktest.update_fill

    def _tracked_update_fill(self_bt, fill):
        orig_update_fill(self_bt, fill)
        recorded_fills.append(
            {
                "instrument": inst_label(fill.order.instrument, cash_label),
                "quantity": pack(float(fill.filled_units)),
                "filled_price": pack(float(fill.filled_price)),
                "execution_time": fmt_ts(fill.order.execution_end_time()),
                "source": str(fill.order.source),
            }
        )

    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "start": fmt_date(start),
        "end": fmt_date(end),
        "week_mask": "1111100",
        "holidays": [fmt_date(h) for h in holidays],
        "initial_value": pack(float(initial_value)),
        "eod_valuation_time": list(eod_valuation_time),
        "cash_label": cash_label,
        "daily_prices": {
            name: {
                "dates": [fmt_date(d) for d in daily_prices[name][0]],
                "values": [pack(float(v)) for v in daily_prices[name][1]],
            }
            for name in sorted(daily_prices.keys())
        },
        "intraday_prices": {
            name: [[fmt_ts(t), pack(float(v))] for t, v in intraday_prices[name]]
            for name in sorted(intraday_prices.keys())
        },
        "orders": [encode_order_spec(o) for o in orders],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pae_mod.is_business_day = fn_is_bday
        pae_mod.business_day_offset = fn_bday_offset
        pae_mod.prev_business_date = fn_prev_bday
        PredefinedAssetBacktest.update_fill = _tracked_update_fill
        try:
            bt = engine.run_backtest(strategy, start=start, end=end, initial_value=initial_value)
        except Exception as exc:
            if rust_error is None:
                raise
            entry["error"] = {
                "rust": rust_error,
                "upstream_type": type(exc).__name__,
            }
        else:
            if rust_error is not None:
                raise AssertionError(f"{case_id}: expected upstream error for {rust_error}, got output")
            perf_dates = [fmt_date(d) for d in bt.performance.index]
            perf_values = [pack(float(v)) for v in bt.performance.values]

            snapshots = []
            for d in bt.historical_holdings.index:
                h_map = bt.historical_holdings[d]
                w_map = bt.historical_weights[d]
                h_out = {
                    inst_label(k, cash_label): pack(float(v))
                    for k, v in sorted(h_map.items(), key=lambda kv: inst_label(kv[0], cash_label))
                }
                w_out = {
                    inst_label(k, cash_label): pack(float(v))
                    for k, v in sorted(w_map.items(), key=lambda kv: inst_label(kv[0], cash_label))
                }
                snapshots.append(
                    {
                        "date": fmt_date(d),
                        "holdings": h_out,
                        "weights": w_out,
                    }
                )

            ledger_df = bt.trade_ledger()
            ledger = []
            for _, row in ledger_df.iterrows():
                status = str(row["Status"])
                close_ts = None if pd.isna(row["Close"]) else fmt_ts(row["Close"])
                close_px = None if pd.isna(row["Close Value"]) else pack(float(row["Close Value"]))
                pnl = None if pd.isna(row["Trade PnL"]) else pack(float(row["Trade PnL"]))
                ledger.append(
                    {
                        "instrument": inst_label(row["Instrument"], cash_label),
                        "open_time": fmt_ts(row["Open"]),
                        "close_time": close_ts,
                        "open_price": pack(float(row["Open Value"])),
                        "close_price": close_px,
                        "status": status,
                        "trade_pnl": pnl,
                    }
                )

            entry["output"] = {
                "performance": {
                    "dates": perf_dates,
                    "values": perf_values,
                },
                "snapshots": snapshots,
                "fills": recorded_fills,
                "trade_ledger": ledger,
            }
        finally:
            pae_mod.is_business_day = orig_is_bday
            pae_mod.business_day_offset = orig_bday_offset
            pae_mod.prev_business_date = orig_prev_bday
            PredefinedAssetBacktest.update_fill = orig_update_fill

    cases.append(entry)


def ts(y, m, d, h=0, mi=0, s=0) -> dt.datetime:
    return dt.datetime(y, m, d, h, mi, s)


# 1. Flagship upstream test_backtest_predefined MOC
add_case(
    "upstream-moc-basic",
    "test_backtest_predefined.py::test_backtest_predefined[MOC]",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "TestRic": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [1.0, 1.5, 2.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "TestRic",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "Test",
        }
    ],
)

# 2. Flagship upstream test_backtest_predefined TWAP (no ON positions)
add_case(
    "upstream-twap-intraday-roundtrip",
    "test_backtest_predefined.py::test_backtest_predefined[TWAP]",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "TestRic": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [1.0, 1.5, 2.0],
        )
    },
    intraday_prices={
        "TestRic": [
            (ts(2021, 1, 5, 10, 30, 0), 16.0),
            (ts(2021, 1, 5, 14, 30, 0), 25.0),
            (ts(2021, 1, 6, 10, 30, 0), 30.0),
            (ts(2021, 1, 6, 14, 30, 0), 40.0),
        ]
    },
    orders=[
        {
            "kind": "TWAP",
            "instrument": "TestRic",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 9, 30, 0),
            "start": ts(2021, 1, 5, 10, 0, 0),
            "end": ts(2021, 1, 5, 10, 30, 0),
            "source": "FuturesExample",
        },
        {
            "kind": "TWAP",
            "instrument": "TestRic",
            "quantity": -1.0,
            "generation_time": ts(2021, 1, 5, 9, 30, 0),
            "start": ts(2021, 1, 5, 14, 0, 0),
            "end": ts(2021, 1, 5, 14, 30, 0),
            "source": "FuturesExample",
        },
        {
            "kind": "TWAP",
            "instrument": "TestRic",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 6, 9, 30, 0),
            "start": ts(2021, 1, 6, 10, 0, 0),
            "end": ts(2021, 1, 6, 10, 30, 0),
            "source": "FuturesExample",
        },
        {
            "kind": "TWAP",
            "instrument": "TestRic",
            "quantity": -1.0,
            "generation_time": ts(2021, 1, 6, 9, 30, 0),
            "start": ts(2021, 1, 6, 14, 0, 0),
            "end": ts(2021, 1, 6, 14, 30, 0),
            "source": "FuturesExample",
        },
    ],
)

# 3. Multi-tick TWAP strictly verifying half-open (start, end] boundary
add_case(
    "twap-half-open-multi-tick",
    "synthetic::twap-half-open",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "FutA": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [10.0, 14.0, 16.0],
        )
    },
    intraday_prices={
        "FutA": [
            (ts(2021, 1, 5, 10, 0, 0), 999.0),  # exact start: EXCLUDED
            (ts(2021, 1, 5, 10, 10, 0), 10.0),
            (ts(2021, 1, 5, 10, 20, 0), 12.0),
            (ts(2021, 1, 5, 10, 30, 0), 14.0),  # exact end: INCLUDED -> mean = 12.0
            (ts(2021, 1, 5, 10, 31, 0), 888.0),  # after end: EXCLUDED
            (ts(2021, 1, 6, 14, 0, 0), 777.0),  # exact start: EXCLUDED
            (ts(2021, 1, 6, 14, 15, 0), 15.0),
            (ts(2021, 1, 6, 14, 30, 0), 19.0),  # exact end: INCLUDED -> mean = 17.0
        ]
    },
    orders=[
        {
            "kind": "TWAP",
            "instrument": "FutA",
            "quantity": 2.0,
            "generation_time": ts(2021, 1, 5, 9, 30, 0),
            "start": ts(2021, 1, 5, 10, 0, 0),
            "end": ts(2021, 1, 5, 10, 30, 0),
            "source": "TwapEntry",
        },
        {
            "kind": "TWAP",
            "instrument": "FutA",
            "quantity": -2.0,
            "generation_time": ts(2021, 1, 6, 9, 30, 0),
            "start": ts(2021, 1, 6, 14, 0, 0),
            "end": ts(2021, 1, 6, 14, 30, 0),
            "source": "TwapExit",
        },
    ],
)

# 4. OrderAtMarket intraday round-trip
add_case(
    "at-market-intraday-roundtrip",
    "synthetic::at-market-intraday",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 5),
    intraday_prices={
        "SpotX": [
            (ts(2021, 1, 5, 10, 15, 0), 20.5),
            (ts(2021, 1, 5, 15, 45, 0), 23.0),
        ]
    },
    orders=[
        {
            "kind": "AT_MARKET",
            "instrument": "SpotX",
            "quantity": 4.0,
            "generation_time": ts(2021, 1, 5, 10, 15, 0),
            "execution_time": ts(2021, 1, 5, 10, 15, 0),
            "source": "MktOpen",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "SpotX",
            "quantity": -4.0,
            "generation_time": ts(2021, 1, 5, 15, 45, 0),
            "execution_time": ts(2021, 1, 5, 15, 45, 0),
            "source": "MktClose",
        },
    ],
)

# 5. OrderAtMarket multi-day hold with daily EOD valuation
add_case(
    "at-market-multi-day-hold",
    "synthetic::at-market-multiday",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 7),
    daily_prices={
        "SpotX": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6), dt.date(2021, 1, 7)],
            [49.0, 52.0, 48.0, 55.0],
        )
    },
    intraday_prices={
        "SpotX": [
            (ts(2021, 1, 5, 9, 45, 0), 50.0),
            (ts(2021, 1, 7, 11, 0, 0), 54.0),
        ]
    },
    orders=[
        {
            "kind": "AT_MARKET",
            "instrument": "SpotX",
            "quantity": 1.5,
            "generation_time": ts(2021, 1, 5, 9, 45, 0),
            "execution_time": ts(2021, 1, 5, 9, 45, 0),
            "source": "Enter",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "SpotX",
            "quantity": -1.5,
            "generation_time": ts(2021, 1, 7, 11, 0, 0),
            "execution_time": ts(2021, 1, 7, 11, 0, 0),
            "source": "Exit",
        },
    ],
)

# 6. OrderCost fee outflow alongside MOC trade
add_case(
    "order-cost-fee-outflow",
    "synthetic::order-cost-fee",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "BondA": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [10.0, 12.0, 15.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "BondA",
            "quantity": 2.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "MocBuy",
        },
        {
            "kind": "COST",
            "instrument": "USD",
            "quantity": -2.5,
            "generation_time": ts(2021, 1, 5, 11, 0, 0),
            "execution_time": ts(2021, 1, 5, 11, 0, 0),
            "source": "Commission",
        },
    ],
)

# 7. OrderCost fee + rebate pair (verifies USD FIFO closed trade ledger with 0.0 PnL)
add_case(
    "order-cost-fee-and-rebate",
    "synthetic::order-cost-pair",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    orders=[
        {
            "kind": "COST",
            "instrument": "USD",
            "quantity": -3.0,
            "generation_time": ts(2021, 1, 5, 12, 0, 0),
            "execution_time": ts(2021, 1, 5, 12, 0, 0),
            "source": "Fee",
        },
        {
            "kind": "COST",
            "instrument": "USD",
            "quantity": 0.75,
            "generation_time": ts(2021, 1, 6, 12, 0, 0),
            "execution_time": ts(2021, 1, 6, 12, 0, 0),
            "source": "Rebate",
        },
    ],
)

# 8. Multi-asset mixed MOC, TWAP, AtMarket, and Cost orders
add_case(
    "multi-asset-mixed-orders",
    "synthetic::multi-asset",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 7),
    initial_value=1000.0,
    daily_prices={
        "AssetA": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6), dt.date(2021, 1, 7)],
            [100.0, 102.0, 108.0, 110.0],
        ),
        "AssetB": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6), dt.date(2021, 1, 7)],
            [50.0, 51.0, 49.0, 53.0],
        ),
    },
    intraday_prices={
        "AssetA": [
            (ts(2021, 1, 6, 10, 15, 0), 106.0),
            (ts(2021, 1, 6, 10, 30, 0), 108.0),
        ],
        "AssetB": [
            (ts(2021, 1, 5, 11, 0, 0), 50.5),
            (ts(2021, 1, 7, 15, 0, 0), 52.5),
        ],
    },
    orders=[
        {
            "kind": "AT_MARKET",
            "instrument": "AssetB",
            "quantity": 4.0,
            "generation_time": ts(2021, 1, 5, 11, 0, 0),
            "execution_time": ts(2021, 1, 5, 11, 0, 0),
            "source": "BuyB",
        },
        {
            "kind": "MOC",
            "instrument": "AssetA",
            "quantity": 2.0,
            "generation_time": ts(2021, 1, 5, 14, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "BuyA",
        },
        {
            "kind": "COST",
            "instrument": "USD",
            "quantity": -1.5,
            "generation_time": ts(2021, 1, 5, 16, 0, 0),
            "execution_time": ts(2021, 1, 5, 16, 0, 0),
            "source": "Fee1",
        },
        {
            "kind": "TWAP",
            "instrument": "AssetA",
            "quantity": -2.0,
            "generation_time": ts(2021, 1, 6, 9, 30, 0),
            "start": ts(2021, 1, 6, 10, 0, 0),
            "end": ts(2021, 1, 6, 10, 30, 0),
            "source": "SellA",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "AssetB",
            "quantity": -4.0,
            "generation_time": ts(2021, 1, 7, 15, 0, 0),
            "execution_time": ts(2021, 1, 7, 15, 0, 0),
            "source": "SellB",
        },
    ],
)

# 9. Short-first profitable round-trip (short open -> long close; unit spread PnL)
add_case(
    "short-first-profit-fifo",
    "synthetic::short-first-profit",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 7),
    initial_value=500.0,
    daily_prices={
        "ShortFut": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6), dt.date(2021, 1, 7)],
            [102.0, 100.0, 95.0, 90.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "ShortFut",
            "quantity": -2.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "ShortEntry",
        },
        {
            "kind": "MOC",
            "instrument": "ShortFut",
            "quantity": 2.0,
            "generation_time": ts(2021, 1, 7, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 7),
            "source": "ShortCover",
        },
    ],
)

# 10. Short-first losing round-trip
add_case(
    "short-first-loss-fifo",
    "synthetic::short-first-loss",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    initial_value=500.0,
    daily_prices={
        "ShortFut": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [50.0, 50.0, 58.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "ShortFut",
            "quantity": -1.5,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "ShortEntry",
        },
        {
            "kind": "MOC",
            "instrument": "ShortFut",
            "quantity": 1.5,
            "generation_time": ts(2021, 1, 6, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 6),
            "source": "ShortCover",
        },
    ],
)

# 11. Multiple sequential round-trips + open remainder (FIFO pairing L1-S1, L2-S2, L3 Open)
add_case(
    "multiple-sequential-roundtrips-fifo",
    "synthetic::multi-roundtrip-fifo",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 8),
    initial_value=1000.0,
    daily_prices={
        "EqA": (
            [
                dt.date(2021, 1, 4),
                dt.date(2021, 1, 5),
                dt.date(2021, 1, 6),
                dt.date(2021, 1, 7),
                dt.date(2021, 1, 8),
            ],
            [10.0, 12.0, 15.0, 14.0, 18.0],
        )
    },
    intraday_prices={
        "EqA": [
            (ts(2021, 1, 5, 10, 0, 0), 11.0),
            (ts(2021, 1, 5, 14, 0, 0), 13.0),
            (ts(2021, 1, 6, 10, 0, 0), 14.5),
            (ts(2021, 1, 7, 11, 0, 0), 16.0),
            (ts(2021, 1, 8, 10, 0, 0), 17.0),
        ]
    },
    orders=[
        {
            "kind": "AT_MARKET",
            "instrument": "EqA",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_time": ts(2021, 1, 5, 10, 0, 0),
            "source": "L1",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "EqA",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 14, 0, 0),
            "execution_time": ts(2021, 1, 5, 14, 0, 0),
            "source": "L2",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "EqA",
            "quantity": -1.0,
            "generation_time": ts(2021, 1, 6, 10, 0, 0),
            "execution_time": ts(2021, 1, 6, 10, 0, 0),
            "source": "S1",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "EqA",
            "quantity": -1.0,
            "generation_time": ts(2021, 1, 7, 11, 0, 0),
            "execution_time": ts(2021, 1, 7, 11, 0, 0),
            "source": "S2",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "EqA",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 8, 10, 0, 0),
            "execution_time": ts(2021, 1, 8, 10, 0, 0),
            "source": "L3",
        },
    ],
)

# 12. Partial size mismatch (buy 3, sell -1 -> order-level FIFO closed pair + 2 units still held)
add_case(
    "partial-size-order-fifo",
    "synthetic::partial-size-fifo",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "FutP": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [10.0, 12.0, 15.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "FutP",
            "quantity": 3.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "Buy3",
        },
        {
            "kind": "MOC",
            "instrument": "FutP",
            "quantity": -1.0,
            "generation_time": ts(2021, 1, 6, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 6),
            "source": "Sell1",
        },
    ],
)

# 13. Same-timestamp tie stability & equal end_time long/short tie (< favors short as open)
add_case(
    "same-timestamp-tie-stability",
    "synthetic::tie-stability",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "TieA": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [20.0, 22.0, 25.0],
        )
    },
    intraday_prices={
        "TieA": [
            (ts(2021, 1, 5, 11, 0, 0), 21.0),
            (ts(2021, 1, 5, 14, 0, 0), 23.0),
        ]
    },
    orders=[
        # Submitted first, later execution_end_time (14:00)
        {
            "kind": "AT_MARKET",
            "instrument": "TieA",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 9, 0, 0),
            "execution_time": ts(2021, 1, 5, 14, 0, 0),
            "source": "LaterEndFirstSub",
        },
        # Submitted second, earlier execution_end_time (11:00) -> fills FIRST, but in ledger longs queue is [LaterEndFirstSub, EarlierEndSecondSub]
        {
            "kind": "AT_MARKET",
            "instrument": "TieA",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 9, 0, 0),
            "execution_time": ts(2021, 1, 5, 11, 0, 0),
            "source": "EarlierEndSecondSub",
        },
        # Short at exact same end_time as LaterEndFirstSub (14:00) -> l.end_time < s.end_time is False -> short is open, long is close
        {
            "kind": "AT_MARKET",
            "instrument": "TieA",
            "quantity": -1.0,
            "generation_time": ts(2021, 1, 5, 9, 0, 0),
            "execution_time": ts(2021, 1, 5, 14, 0, 0),
            "source": "SameEndShort",
        },
    ],
)

# 14. Holiday calendar skipping (mid-week holiday 2021-01-06 skipped; order on holiday ignored)
add_case(
    "holiday-calendar-skip",
    "synthetic::holiday-skip",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 8),
    holidays=[dt.date(2021, 1, 6)],
    daily_prices={
        "HolFut": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 7), dt.date(2021, 1, 8)],
            [10.0, 11.0, 13.0, 14.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "HolFut",
            "quantity": 2.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "BeforeHol",
        },
        # Generated on holiday 2021-01-06 -> NOT an active business day, never triggered!
        {
            "kind": "MOC",
            "instrument": "HolFut",
            "quantity": 5.0,
            "generation_time": ts(2021, 1, 6, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 6),
            "source": "OnHolidayIgnored",
        },
        {
            "kind": "MOC",
            "instrument": "HolFut",
            "quantity": -2.0,
            "generation_time": ts(2021, 1, 8, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 8),
            "source": "AfterHol",
        },
    ],
)

# 15. Weekend start & end backward adjustment + holiday on rolled start
add_case(
    "weekend-start-end-backward-roll",
    "synthetic::weekend-roll",
    dt.date(2021, 1, 3),  # Sunday -> Fri 2021-01-01 (holiday!) -> Thu 2020-12-31
    dt.date(2021, 1, 9),  # Saturday -> Fri 2021-01-08
    holidays=[dt.date(2021, 1, 1)],
    daily_prices={
        "RollFut": (
            [
                dt.date(2020, 12, 31),
                dt.date(2021, 1, 4),
                dt.date(2021, 1, 5),
                dt.date(2021, 1, 6),
                dt.date(2021, 1, 7),
                dt.date(2021, 1, 8),
            ],
            [100.0, 101.0, 103.0, 102.0, 105.0, 106.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "RollFut",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 4, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 4),
            "source": "MondayEntry",
        }
    ],
)

# 16. start == end (initial NAV only, zero active days)
add_case(
    "start-equals-end-initial-only",
    "synthetic::start-eq-end",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 4),
    initial_value=1234.5,
    orders=[
        {
            "kind": "MOC",
            "instrument": "IgnoredFut",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 4, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 4),
            "source": "OnStartDayIgnored",
        }
    ],
)

# 17. Friday start & Saturday end -> adjusted_start == adjusted_end (initial NAV only)
add_case(
    "friday-to-saturday-collapses-to-initial",
    "synthetic::fri-sat-collapse",
    dt.date(2021, 1, 8),  # Friday
    dt.date(2021, 1, 9),  # Saturday -> rolls back to Friday 2021-01-08
    initial_value=750.0,
)

# 18. Custom initial_value and zero-order hold across 4 business days
add_case(
    "custom-initial-zero-orders",
    "synthetic::custom-initial-hold",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 8),
    initial_value=250000.0,
)

# 19. Epsilon 1e-12 zero-position pruning vs retention
add_case(
    "epsilon-zero-position-pruning",
    "synthetic::epsilon-pruning",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "Pruned": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [10.0, 10.0, 20.0],
        ),
        "Kept": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)],
            [10.0, 10.0, 20.0],
        ),
    },
    intraday_prices={
        "Pruned": [
            (ts(2021, 1, 5, 10, 0, 0), 10.0),
            (ts(2021, 1, 5, 14, 0, 0), 12.0),
        ],
        "Kept": [
            (ts(2021, 1, 5, 10, 0, 0), 10.0),
            (ts(2021, 1, 5, 14, 0, 0), 12.0),
        ],
    },
    orders=[
        # Pruned: +1.0 then -1.0 + 5e-13 -> |residual| = 5e-13 <= 1e-12 -> pruned!
        {
            "kind": "AT_MARKET",
            "instrument": "Pruned",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_time": ts(2021, 1, 5, 10, 0, 0),
            "source": "P1",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "Pruned",
            "quantity": -1.0 + 5e-13,
            "generation_time": ts(2021, 1, 5, 14, 0, 0),
            "execution_time": ts(2021, 1, 5, 14, 0, 0),
            "source": "P2",
        },
        # Kept: +1.0 then -1.0 + 5e-12 -> |residual| = 5e-12 > 1e-12 -> retained!
        {
            "kind": "AT_MARKET",
            "instrument": "Kept",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_time": ts(2021, 1, 5, 10, 0, 0),
            "source": "K1",
        },
        {
            "kind": "AT_MARKET",
            "instrument": "Kept",
            "quantity": -1.0 + 5e-12,
            "generation_time": ts(2021, 1, 5, 14, 0, 0),
            "execution_time": ts(2021, 1, 5, 14, 0, 0),
            "source": "K2",
        },
    ],
)

# 20. Multi-day deferred MOC (generated on Jan 5, execution_date Jan 6)
add_case(
    "deferred-next-day-moc",
    "synthetic::deferred-moc",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 7),
    daily_prices={
        "DefFut": (
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6), dt.date(2021, 1, 7)],
            [10.0, 11.0, 12.0, 15.0],
        )
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "DefFut",
            "quantity": 2.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 6),
            "source": "DeferredMoc",
        }
    ],
)

# 21. Error: Missing daily close for MOC execution
add_case(
    "err-missing-daily-moc",
    "synthetic::error-missing-daily-moc",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "MissMoc": ([dt.date(2021, 1, 4), dt.date(2021, 1, 6)], [10.0, 12.0])
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "MissMoc",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "BadMoc",
        }
    ],
    rust_error="Length",
)

# 22. Error: Empty TWAP half-open window (tick only at exact start -> excluded -> empty -> NaN -> RuntimeError)
add_case(
    "err-empty-twap-window",
    "synthetic::error-empty-twap",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 5),
    intraday_prices={
        "EmptyTwap": [(ts(2021, 1, 5, 10, 0, 0), 25.0)]
    },
    orders=[
        {
            "kind": "TWAP",
            "instrument": "EmptyTwap",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 9, 30, 0),
            "start": ts(2021, 1, 5, 10, 0, 0),
            "end": ts(2021, 1, 5, 10, 30, 0),
            "source": "BadTwap",
        }
    ],
    rust_error="Length",
)

# 23. Error: NaN tick inside TWAP window
add_case(
    "err-nan-twap-tick",
    "synthetic::error-nan-twap",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 5),
    intraday_prices={
        "NanTwap": [
            (ts(2021, 1, 5, 10, 10, 0), 20.0),
            (ts(2021, 1, 5, 10, 20, 0), NAN),
            (ts(2021, 1, 5, 10, 30, 0), 22.0),
        ]
    },
    orders=[
        {
            "kind": "TWAP",
            "instrument": "NanTwap",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 9, 30, 0),
            "start": ts(2021, 1, 5, 10, 0, 0),
            "end": ts(2021, 1, 5, 10, 30, 0),
            "source": "NanTwapOrder",
        }
    ],
    rust_error="Length",
)

# 24. Error: Missing exact intraday tick for OrderAtMarket
add_case(
    "err-missing-intraday-at-market",
    "synthetic::error-missing-at-market",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 5),
    intraday_prices={
        "MissMkt": [(ts(2021, 1, 5, 10, 0, 0), 20.0)]
    },
    orders=[
        {
            "kind": "AT_MARKET",
            "instrument": "MissMkt",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 10, 15, 0),
            "execution_time": ts(2021, 1, 5, 10, 15, 0),
            "source": "BadMkt",
        }
    ],
    rust_error="Length",
)

# 25. Error: Missing daily close on EOD mark-to-market valuation
add_case(
    "err-missing-daily-valuation",
    "synthetic::error-missing-eod-val",
    dt.date(2021, 1, 4),
    dt.date(2021, 1, 6),
    daily_prices={
        "MissVal": ([dt.date(2021, 1, 4), dt.date(2021, 1, 5)], [10.0, 12.0])
    },
    orders=[
        {
            "kind": "MOC",
            "instrument": "MissVal",
            "quantity": 1.0,
            "generation_time": ts(2021, 1, 5, 10, 0, 0),
            "execution_date": dt.date(2021, 1, 5),
            "source": "HoldIntoMissingDay",
        }
    ],
    rust_error="Length",
)


payload = {
    "schema": "gsquant-predefined-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "cases": cases,
}

out_path = ROOT / "predefined-vectors.json"
out_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {len(cases)} predefined cases to {out_path}")
