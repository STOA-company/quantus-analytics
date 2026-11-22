"""Replay frozen upstream risk aggregation, subtraction, sorting & point_sort_order vectors without importing Python upstream or accessing the network."""
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

raw = (ROOT / "risk-vectors.json").read_bytes()
data = json.loads(raw)
assert data["schema"] == "gsquant-risk-parity-v1"
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


def encode_table_lines(table_spec) -> list:
    rows = table_spec["rows"]
    unit = table_spec["unit"]
    if unit is None:
        header = f"TABLE\t{len(rows)}\tNONE"
    else:
        header = f"TABLE\t{len(rows)}\tSOME\t{unit}"
    lines = [header]
    for r in rows:
        d_str = "-" if r["date"] is None else r["date"]
        t_str = "-" if r["time"] is None else f"T:{r['time']}"
        lines.append(
            "\t".join(
                [
                    d_str,
                    t_str,
                    r["mkt_type"],
                    r["mkt_asset"],
                    r["mkt_class"],
                    r["mkt_point"],
                    unpack_str(r["value"]),
                ]
            )
        )
    return lines


def parse_table_output(lines):
    assert lines, "empty replay output"
    if len(lines) == 1 and lines[0].startswith("ERROR "):
        return {"error": lines[0].split(" ", 1)[1]}
    unit_parts = lines[0].split("\t")
    assert unit_parts[0] == "UNIT", lines[0]
    if unit_parts[1] == "NONE":
        unit = None
    elif unit_parts[1] == "SOME":
        unit = unit_parts[2] if len(unit_parts) > 2 else ""
    else:
        raise AssertionError(f"unknown UNIT tag: {lines[0]}")
    rows = []
    for line in lines[1:]:
        f = line.split("\t")
        assert len(f) == 7, (len(f), line)
        date_val = None if f[0] == "-" else f[0]
        time_val = None if f[1] == "-" else f[1][2:]
        rows.append(
            {
                "date": date_val,
                "time": time_val,
                "mkt_type": f[2],
                "mkt_asset": f[3],
                "mkt_class": f[4],
                "mkt_point": f[5],
                "value": float(f[6]),
            }
        )
    return {"unit": unit, "rows": rows}


def compare_float(case_id, label, actual: float, raw_target, stats):
    target = unpack_float(raw_target)
    if math.isnan(target):
        assert math.isnan(actual), (case_id, label, actual)
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


def compare_tables(case_id, actual, expected, stats):
    assert "error" not in actual, (case_id, actual)
    assert actual["unit"] == expected["unit"], (case_id, actual["unit"], expected["unit"])
    act_rows = actual["rows"]
    exp_rows = expected["rows"]
    assert len(act_rows) == len(exp_rows), (case_id, len(act_rows), len(exp_rows))
    for i, (ar, er) in enumerate(zip(act_rows, exp_rows)):
        for k in ("date", "time", "mkt_type", "mkt_asset", "mkt_class", "mkt_point"):
            assert ar[k] == er[k], (case_id, i, k, ar[k], er[k])
        compare_float(case_id, f"row[{i}].value", ar["value"], er["value"], stats)


stats = {"max_abs": 0.0, "max_rel": 0.0}
point_cases = 0
sort_cases = 0
agg_cases = 0
sub_cases = 0
error_cases = 0

for case in data["cases"]:
    op = case["operation"]
    ref_date = case["ref_date"]

    if op == "point_sort_order":
        payload_in = f"point_sort_order {ref_date}\n{case['point']}\n"
    elif op == "sort_risk":
        lines_in = [f"sort_risk {ref_date}"] + encode_table_lines(case["table"])
        payload_in = "\n".join(lines_in) + "\n"
    elif op == "aggregate_risk":
        th = "none" if case["threshold"] is None else unpack_str(case["threshold"])
        tables = case["tables"]
        lines_in = [f"aggregate_risk {ref_date} {th} {len(tables)}"]
        for t in tables:
            lines_in.extend(encode_table_lines(t))
        payload_in = "\n".join(lines_in) + "\n"
    elif op == "subtract_risk":
        lines_in = (
            [f"subtract_risk {ref_date}"]
            + encode_table_lines(case["left"])
            + encode_table_lines(case["right"])
        )
        payload_in = "\n".join(lines_in) + "\n"
    else:
        raise AssertionError(f"unknown operation: {op}")

    stdout_lines = subprocess.run(
        [binary],
        input=payload_in,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    if "error" in case:
        error_cases += 1
        expected_line = f"ERROR {case['error']['rust']}"
        assert stdout_lines == [expected_line], (case["id"], stdout_lines, [expected_line])
        continue

    if op == "point_sort_order":
        point_cases += 1
        assert len(stdout_lines) == 1, (case["id"], stdout_lines)
        exp = case["output"]
        if exp is None:
            assert stdout_lines[0] == "NONE", (case["id"], stdout_lines)
        else:
            assert stdout_lines[0] != "NONE", (case["id"], stdout_lines, exp)
            compare_float(case["id"], "point_sort_order", float(stdout_lines[0]), exp, stats)
    elif op == "sort_risk":
        sort_cases += 1
        compare_tables(case["id"], parse_table_output(stdout_lines), case["output"], stats)
    elif op == "aggregate_risk":
        agg_cases += 1
        compare_tables(case["id"], parse_table_output(stdout_lines), case["output"], stats)
    elif op == "subtract_risk":
        sub_cases += 1
        assert case.get("upstream_note") == "upstream_subtract_risk_columns_names_bug", case["id"]
        compare_tables(case["id"], parse_table_output(stdout_lines), case["output"], stats)

print(
    f"{len(data['cases'])} risk cases passed ({point_cases} point_sort_order, "
    f"{sort_cases} sort_risk, {agg_cases} aggregate_risk, {sub_cases} subtract_risk, "
    f"{error_cases} error); "
    f"max absolute error {stats['max_abs']:.3g}; max relative error {stats['max_rel']:.3g}"
)
