"""Replay frozen upstream PredefinedAssetEngine vectors without importing Python upstream or accessing the network."""
import argparse
import hashlib
import json
import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--replay", action="store_true", required=True)
parser.parse_args()

raw = (ROOT / "predefined-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-predefined-parity-v1"
assert data["upstream_sha"] == "70220e300524d54bd273be0f4c485d53b3780482"
assert (
    hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest()
    == data["python_lock_sha256"]
)

subprocess.run(
    ["cargo", "build", "--offline", "--locked", "--example", "replay"],
    cwd=ROOT.parent,
    check=True,
)
binary = str(ROOT.parent / "target/debug/examples/replay")

ATOL = 1e-12
RTOL = 1e-10


def unpack_str(v) -> str:
    if v == "NaN":
        return "NaN"
    if v == "Infinity":
        return "inf"
    if v == "-Infinity":
        return "-inf"
    return repr(float(v))


def unpack_float(v) -> float:
    if v == "NaN":
        return float("nan")
    if v == "Infinity":
        return float("inf")
    if v == "-Infinity":
        return float("-inf")
    return float(v)


def compare_float(case_id, label, actual: float, expected_raw, stats):
    target = unpack_float(expected_raw)
    if math.isnan(target):
        assert math.isnan(actual), (case_id, label, actual, target)
    elif math.isinf(target):
        assert actual == target, (case_id, label, actual, target)
    else:
        delta = abs(actual - target)
        assert math.isfinite(actual) and delta <= ATOL + RTOL * abs(target), (
            case_id,
            label,
            actual,
            target,
        )
        stats["max_abs"] = max(stats["max_abs"], delta)
        if target != 0.0:
            stats["max_rel"] = max(stats["max_rel"], delta / abs(target))


def encode_case(case) -> str:
    holidays = case["holidays"]
    daily = case["daily_prices"]
    intraday = case["intraday_prices"]
    orders = case["orders"]
    eod_h, eod_m, eod_s = case["eod_valuation_time"]

    lines = [
        f"predefined_backtest {case['start']} {case['end']} {case['week_mask']} "
        f"{eod_h} {eod_m} {eod_s} {unpack_str(case['initial_value'])} {case['cash_label']} "
        f"{len(holidays)} {len(daily)} {len(intraday)} {len(orders)}"
    ]
    lines.extend(holidays)

    for inst in sorted(daily.keys()):
        spec = daily[inst]
        lines.append(f"DAILY {inst} {len(spec['dates'])}")
        for d, v in zip(spec["dates"], spec["values"]):
            lines.append(f"{d} {unpack_str(v)}")

    for inst in sorted(intraday.keys()):
        ticks = intraday[inst]
        lines.append(f"INTRADAY {inst} {len(ticks)}")
        for ts_str, v in ticks:
            lines.append(f"{ts_str} {unpack_str(v)}")

    for o in orders:
        kind = o["kind"]
        qty = unpack_str(o["quantity"])
        if kind == "MOC":
            lines.append(
                f"MOC\t{o['instrument']}\t{qty}\t{o['generation_time']}\t{o['execution_date']}\t{o['source']}"
            )
        elif kind == "TWAP":
            lines.append(
                f"TWAP\t{o['instrument']}\t{qty}\t{o['generation_time']}\t{o['start']}\t{o['end']}\t{o['source']}"
            )
        elif kind == "AT_MARKET":
            lines.append(
                f"AT_MARKET\t{o['instrument']}\t{qty}\t{o['generation_time']}\t{o['execution_time']}\t{o['source']}"
            )
        elif kind == "COST":
            lines.append(
                f"COST\t{o['instrument']}\t{qty}\t{o['generation_time']}\t{o['execution_time']}\t{o['source']}"
            )
        else:
            raise AssertionError(f"unknown order kind: {kind}")

    return "\n".join(lines) + "\n"


def parse_output(lines):
    assert lines, "empty replay output"
    if len(lines) == 1 and lines[0].startswith("ERROR "):
        return {"error": lines[0].split(" ", 1)[1]}

    idx = 0
    tag, n_perf_s = lines[idx].split()
    assert tag == "PERFORMANCE", lines[idx]
    n_perf = int(n_perf_s)
    idx += 1
    perf_dates, perf_vals = [], []
    for _ in range(n_perf):
        d_s, v_s = lines[idx].split()
        perf_dates.append(d_s)
        perf_vals.append(float(v_s))
        idx += 1

    tag, n_snap_s = lines[idx].split()
    assert tag == "SNAPSHOTS", lines[idx]
    n_snap = int(n_snap_s)
    idx += 1
    snapshots = []
    for _ in range(n_snap):
        stag, d_s, n_ent_s = lines[idx].split()
        assert stag == "SNAP", lines[idx]
        n_ent = int(n_ent_s)
        idx += 1
        holdings, weights = {}, {}
        for _ in range(n_ent):
            inst, h_s, w_s = lines[idx].split("\t")
            holdings[inst] = float(h_s)
            weights[inst] = float(w_s)
            idx += 1
        snapshots.append({"date": d_s, "holdings": holdings, "weights": weights})

    tag, n_fills_s = lines[idx].split()
    assert tag == "FILLS", lines[idx]
    n_fills = int(n_fills_s)
    idx += 1
    fills = []
    for _ in range(n_fills):
        inst, qty_s, px_s, ex_ts, src = lines[idx].split("\t")
        fills.append(
            {
                "instrument": inst,
                "quantity": float(qty_s),
                "filled_price": float(px_s),
                "execution_time": ex_ts,
                "source": src,
            }
        )
        idx += 1

    tag, n_led_s = lines[idx].split()
    assert tag == "LEDGER", lines[idx]
    n_led = int(n_led_s)
    idx += 1
    ledger = []
    for _ in range(n_led):
        inst, open_ts, close_ts, open_px_s, close_px_s, status, pnl_s = lines[idx].split("\t")
        ledger.append(
            {
                "instrument": inst,
                "open_time": open_ts,
                "close_time": None if close_ts == "-" else close_ts,
                "open_price": float(open_px_s),
                "close_price": None if close_px_s == "-" else float(close_px_s),
                "status": status,
                "trade_pnl": None if pnl_s == "-" else float(pnl_s),
            }
        )
        idx += 1

    assert idx == len(lines), (idx, len(lines))
    return {
        "performance": {"dates": perf_dates, "values": perf_vals},
        "snapshots": snapshots,
        "fills": fills,
        "trade_ledger": ledger,
    }


stats = {"max_abs": 0.0, "max_rel": 0.0}
value_cases = 0
error_cases = 0

for case in data["cases"]:
    cid = case["id"]
    stdout_lines = subprocess.run(
        [binary],
        input=encode_case(case),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    if "error" in case:
        error_cases += 1
        expected_line = f"ERROR {case['error']['rust']}"
        assert stdout_lines == [expected_line], (cid, stdout_lines, [expected_line])
        continue

    value_cases += 1
    actual = parse_output(stdout_lines)
    assert "error" not in actual, (cid, actual)
    exp = case["output"]

    # 1. Performance series
    assert actual["performance"]["dates"] == exp["performance"]["dates"], (
        cid,
        actual["performance"]["dates"],
        exp["performance"]["dates"],
    )
    assert len(actual["performance"]["values"]) == len(exp["performance"]["values"]), cid
    for i, (av, ev) in enumerate(zip(actual["performance"]["values"], exp["performance"]["values"])):
        compare_float(cid, f"performance[{i}]", av, ev, stats)

    # 2. Daily EOD snapshots (dates, holdings, weights)
    assert len(actual["snapshots"]) == len(exp["snapshots"]), (
        cid,
        len(actual["snapshots"]),
        len(exp["snapshots"]),
    )
    for i, (asnap, esnap) in enumerate(zip(actual["snapshots"], exp["snapshots"])):
        assert asnap["date"] == esnap["date"], (cid, i, asnap["date"], esnap["date"])
        assert sorted(asnap["holdings"].keys()) == sorted(esnap["holdings"].keys()), (
            cid,
            i,
            asnap["holdings"],
            esnap["holdings"],
        )
        assert sorted(asnap["weights"].keys()) == sorted(esnap["weights"].keys()), (
            cid,
            i,
            asnap["weights"],
            esnap["weights"],
        )
        for k in esnap["holdings"]:
            compare_float(cid, f"snapshot[{i}].holdings[{k}]", asnap["holdings"][k], esnap["holdings"][k], stats)
            compare_float(cid, f"snapshot[{i}].weights[{k}]", asnap["weights"][k], esnap["weights"][k], stats)

    # 3. Chronological fills
    assert len(actual["fills"]) == len(exp["fills"]), (cid, actual["fills"], exp["fills"])
    for i, (af, ef) in enumerate(zip(actual["fills"], exp["fills"])):
        assert af["instrument"] == ef["instrument"], (cid, f"fill[{i}].instrument", af, ef)
        assert af["execution_time"] == ef["execution_time"], (cid, f"fill[{i}].execution_time", af, ef)
        assert af["source"] == ef["source"], (cid, f"fill[{i}].source", af, ef)
        compare_float(cid, f"fill[{i}].quantity", af["quantity"], ef["quantity"], stats)
        compare_float(cid, f"fill[{i}].filled_price", af["filled_price"], ef["filled_price"], stats)

    # 4. FIFO trade_ledger
    assert len(actual["trade_ledger"]) == len(exp["trade_ledger"]), (
        cid,
        actual["trade_ledger"],
        exp["trade_ledger"],
    )
    for i, (al, el) in enumerate(zip(actual["trade_ledger"], exp["trade_ledger"])):
        assert al["instrument"] == el["instrument"], (cid, f"ledger[{i}].instrument", al, el)
        assert al["open_time"] == el["open_time"], (cid, f"ledger[{i}].open_time", al, el)
        assert al["close_time"] == el["close_time"], (cid, f"ledger[{i}].close_time", al, el)
        assert al["status"] == el["status"], (cid, f"ledger[{i}].status", al, el)
        compare_float(cid, f"ledger[{i}].open_price", al["open_price"], el["open_price"], stats)
        if el["close_price"] is None:
            assert al["close_price"] is None, (cid, f"ledger[{i}].close_price", al, el)
        else:
            assert al["close_price"] is not None, (cid, f"ledger[{i}].close_price", al, el)
            compare_float(cid, f"ledger[{i}].close_price", al["close_price"], el["close_price"], stats)
        if el["trade_pnl"] is None:
            assert al["trade_pnl"] is None, (cid, f"ledger[{i}].trade_pnl", al, el)
        else:
            assert al["trade_pnl"] is not None, (cid, f"ledger[{i}].trade_pnl", al, el)
            compare_float(cid, f"ledger[{i}].trade_pnl", al["trade_pnl"], el["trade_pnl"], stats)

print(
    f"{len(data['cases'])} predefined cases passed ({value_cases} value, {error_cases} error); "
    f"max absolute error {stats['max_abs']:.3g}; max relative error {stats['max_rel']:.3g}"
)
