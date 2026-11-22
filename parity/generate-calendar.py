"""Generate upstream calendar & day-count parity vectors only. Run with the locked upstream Python environment."""
import datetime as dt
import hashlib
import json
import pathlib
import sys
import numpy as np

# The caller supplies the fixed checkout; no network or Rust execution here.
sys.path.insert(0, sys.argv[1])
from gs_quant.datetime.date import (
    DayCountConvention,
    PaymentFrequency,
    day_count_fraction,
    has_feb_29,
)

ROOT = pathlib.Path(__file__).resolve().parent

CONVENTIONS = {
    "actual_360": DayCountConvention.ACTUAL_360,
    "actual_364": DayCountConvention.ACTUAL_364,
    "actual_36525": DayCountConvention.ACTUAL_365_25,
    "actual_365f": DayCountConvention.ACTUAL_365F,
    "actual_365l": DayCountConvention.ACTUAL_365L,
    "one_one": DayCountConvention.ONE_ONE,
}

FREQUENCIES = {
    "daily": PaymentFrequency.DAILY,
    "weekly": PaymentFrequency.WEEKLY,
    "semi_monthly": PaymentFrequency.SEMI_MONTHLY,
    "monthly": PaymentFrequency.MONTHLY,
    "semi_quarterly": PaymentFrequency.SEMI_QUARTERLY,
    "quarterly": PaymentFrequency.QUARTERLY,
    "tri_annually": PaymentFrequency.TRI_ANNUALLY,
    "semi_annually": PaymentFrequency.SEMI_ANNUALLY,
    "annually": PaymentFrequency.ANNUALLY,
}

ROLL_TO_NUMPY = {
    "raise": "raise",
    "forward": "forward",
    "backward": "backward",
    "modified_following": "modifiedfollowing",
    "modified_preceding": "modifiedpreceding",
}

WEEKDAYS_MASK = "1111100"
SUN_THU_MASK = "1111001"
SAT_ONLY_MASK = "0000010"
ALL_DAYS_MASK = "1111111"
ALL_FALSE_MASK = "0000000"

cases = []


def iso(d: dt.date) -> str:
    return d.isoformat()


def add_has_feb_29(case_id: str, source_ref: str, start: dt.date, end: dt.date):
    out = bool(has_feb_29(start, end))
    cases.append(
        {
            "id": case_id,
            "source_ref": source_ref,
            "operation": "has_feb_29",
            "args": [iso(start), iso(end)],
            "holidays": [],
            "output": {"kind": "bool", "value": out},
        }
    )


def add_dcf(
    case_id: str,
    source_ref: str,
    start: dt.date,
    end: dt.date,
    conv_name: str,
    freq_name: str,
):
    out = float(day_count_fraction(start, end, CONVENTIONS[conv_name], FREQUENCIES[freq_name]))
    cases.append(
        {
            "id": case_id,
            "source_ref": source_ref,
            "operation": "day_count_fraction",
            "args": [iso(start), iso(end), conv_name, freq_name],
            "holidays": [],
            "output": {"kind": "float", "value": out},
        }
    )


def add_is_business_day(
    case_id: str,
    source_ref: str,
    date: dt.date,
    week_mask: str,
    holidays: list,
    rust_error=None,
):
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "is_business_day",
        "args": [iso(date), week_mask],
        "holidays": [iso(h) for h in holidays],
    }
    try:
        cal = np.busdaycalendar(weekmask=week_mask, holidays=holidays)
        out = bool(np.is_busday(date, busdaycal=cal))
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
            raise AssertionError(f"{case_id}: expected upstream error for {rust_error}, got {out}")
        entry["output"] = {"kind": "bool", "value": out}
    cases.append(entry)


def add_business_day_offset(
    case_id: str,
    source_ref: str,
    date: dt.date,
    offset: int,
    roll_name: str,
    week_mask: str,
    holidays: list,
    rust_error=None,
    documented_diff=None,
    expected_rust_error=None,
):
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "business_day_offset",
        "args": [iso(date), offset, roll_name, week_mask],
        "holidays": [iso(h) for h in holidays],
    }
    try:
        cal = np.busdaycalendar(weekmask=week_mask, holidays=holidays)
        out = str(np.busday_offset(date, offset, roll=ROLL_TO_NUMPY[roll_name], busdaycal=cal))
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
            raise AssertionError(f"{case_id}: expected upstream error for {rust_error}, got {out}")
        entry["output"] = {"kind": "date", "value": out}
        if documented_diff is not None:
            entry["documented_diff"] = documented_diff
            entry["expected_rust_error"] = expected_rust_error
    cases.append(entry)


def add_business_day_count(
    case_id: str,
    source_ref: str,
    begin: dt.date,
    end: dt.date,
    week_mask: str,
    holidays: list,
    rust_error=None,
):
    entry = {
        "id": case_id,
        "source_ref": source_ref,
        "operation": "business_day_count",
        "args": [iso(begin), iso(end), week_mask],
        "holidays": [iso(h) for h in holidays],
    }
    try:
        cal = np.busdaycalendar(weekmask=week_mask, holidays=holidays)
        out = int(np.busday_count(begin, end, busdaycal=cal))
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
            raise AssertionError(f"{case_id}: expected upstream error for {rust_error}, got {out}")
        entry["output"] = {"kind": "int", "value": out}
    cases.append(entry)


# 1. has_feb_29 (16 cases)
add_has_feb_29("feb29-within-leap-year", "test_date.py::test_has_feb_29", dt.date(2020, 1, 1), dt.date(2020, 3, 1))
add_has_feb_29("feb29-end-inclusive", "test_date.py::test_has_feb_29", dt.date(2020, 2, 28), dt.date(2020, 2, 29))
add_has_feb_29("feb29-start-exclusive", "test_date.py::test_has_feb_29", dt.date(2020, 2, 29), dt.date(2020, 3, 1))
add_has_feb_29("feb29-same-date-leap-day", "test_date.py::test_has_feb_29", dt.date(2020, 2, 29), dt.date(2020, 2, 29))
add_has_feb_29("feb29-non-leap-year", "test_date.py::test_has_feb_29", dt.date(2019, 1, 1), dt.date(2019, 12, 31))
add_has_feb_29("feb29-cross-year-before", "synthetic-feb29", dt.date(2019, 3, 1), dt.date(2020, 2, 28))
add_has_feb_29("feb29-cross-year-on", "synthetic-feb29", dt.date(2019, 3, 1), dt.date(2020, 2, 29))
add_has_feb_29("feb29-cross-year-after", "synthetic-feb29", dt.date(2019, 3, 1), dt.date(2020, 3, 1))
add_has_feb_29("feb29-after-leap-to-next-year", "synthetic-feb29", dt.date(2020, 2, 29), dt.date(2021, 1, 1))
add_has_feb_29("feb29-century-1900-non-leap", "synthetic-feb29", dt.date(1900, 1, 1), dt.date(1900, 12, 31))
add_has_feb_29("feb29-century-2000-leap", "synthetic-feb29", dt.date(2000, 1, 1), dt.date(2000, 12, 31))
add_has_feb_29("feb29-century-2100-non-leap", "synthetic-feb29", dt.date(2100, 1, 1), dt.date(2100, 12, 31))
add_has_feb_29("feb29-century-2400-leap", "synthetic-feb29", dt.date(2400, 1, 1), dt.date(2400, 12, 31))
add_has_feb_29("feb29-8yr-century-gap-none", "synthetic-feb29", dt.date(1896, 2, 29), dt.date(1904, 2, 28))
add_has_feb_29("feb29-reversed-across-leap", "synthetic-feb29", dt.date(2020, 3, 1), dt.date(2020, 2, 28))
add_has_feb_29("feb29-full-gregorian-span", "synthetic-feb29", dt.date(1, 1, 1), dt.date(9999, 12, 31))

# 2. day_count_fraction (20 cases)
add_dcf("dcf-act360-leap-span", "test_date.py::test_day_count_fraction", dt.date(2020, 1, 1), dt.date(2020, 7, 1), "actual_360", "annually")
add_dcf("dcf-act364-leap-span", "test_date.py::test_day_count_fraction", dt.date(2020, 1, 1), dt.date(2020, 7, 1), "actual_364", "annually")
add_dcf("dcf-act36525-leap-span", "test_date.py::test_day_count_fraction", dt.date(2020, 1, 1), dt.date(2020, 7, 1), "actual_36525", "annually")
add_dcf("dcf-act365f-leap-span", "test_date.py::test_day_count_fraction", dt.date(2020, 1, 1), dt.date(2020, 7, 1), "actual_365f", "annually")
add_dcf("dcf-act365l-annually-has-feb29", "test_date.py::test_day_count_fraction", dt.date(2020, 1, 1), dt.date(2020, 7, 1), "actual_365l", "annually")
add_dcf("dcf-one-one-leap-span", "test_date.py::test_day_count_fraction", dt.date(2020, 1, 1), dt.date(2020, 7, 1), "one_one", "annually")
add_dcf("dcf-act365l-annually-no-feb29", "synthetic-dcf", dt.date(2020, 3, 1), dt.date(2020, 9, 1), "actual_365l", "annually")
add_dcf("dcf-act365l-monthly-leap-end-no-feb29", "synthetic-dcf", dt.date(2020, 3, 1), dt.date(2020, 9, 1), "actual_365l", "monthly")
add_dcf("dcf-act365l-quarterly-nonleap-end-with-feb29", "synthetic-dcf", dt.date(2020, 1, 1), dt.date(2021, 1, 1), "actual_365l", "quarterly")
add_dcf("dcf-act365l-annually-nonleap-end-with-feb29", "synthetic-dcf", dt.date(2020, 1, 1), dt.date(2021, 1, 1), "actual_365l", "annually")
add_dcf("dcf-act365l-daily-leap-end", "synthetic-dcf", dt.date(2024, 1, 15), dt.date(2024, 2, 15), "actual_365l", "daily")
add_dcf("dcf-act365l-weekly-nonleap-end", "synthetic-dcf", dt.date(2023, 1, 15), dt.date(2023, 2, 15), "actual_365l", "weekly")
add_dcf("dcf-act365l-semi-monthly-leap-end", "synthetic-dcf", dt.date(2020, 6, 1), dt.date(2020, 6, 15), "actual_365l", "semi_monthly")
add_dcf("dcf-act365l-semi-quarterly-nonleap", "synthetic-dcf", dt.date(2021, 1, 1), dt.date(2021, 3, 1), "actual_365l", "semi_quarterly")
add_dcf("dcf-act365l-tri-annually-leap", "synthetic-dcf", dt.date(2020, 5, 1), dt.date(2020, 9, 1), "actual_365l", "tri_annually")
add_dcf("dcf-act365l-semi-annually-leap-end-2000", "synthetic-dcf", dt.date(1999, 7, 1), dt.date(2000, 1, 1), "actual_365l", "semi_annually")
add_dcf("dcf-zero-span-act360", "synthetic-dcf", dt.date(2020, 6, 1), dt.date(2020, 6, 1), "actual_360", "annually")
add_dcf("dcf-zero-span-one-one", "synthetic-dcf", dt.date(2020, 6, 1), dt.date(2020, 6, 1), "one_one", "annually")
add_dcf("dcf-negative-act365l-annually", "synthetic-dcf", dt.date(2020, 3, 1), dt.date(2020, 1, 1), "actual_365l", "annually")
add_dcf("dcf-negative-one-one", "synthetic-dcf", dt.date(2020, 3, 1), dt.date(2020, 1, 1), "one_one", "annually")

# 3. is_business_day (12 cases)
NY_2020 = [dt.date(2020, 1, 1)]
add_is_business_day("is-bday-weekday", "synthetic-is-bday", dt.date(2020, 1, 2), WEEKDAYS_MASK, NY_2020)
add_is_business_day("is-bday-holiday", "synthetic-is-bday", dt.date(2020, 1, 1), WEEKDAYS_MASK, NY_2020)
add_is_business_day("is-bday-saturday", "synthetic-is-bday", dt.date(2020, 1, 4), WEEKDAYS_MASK, [])
add_is_business_day("is-bday-sunday", "synthetic-is-bday", dt.date(2020, 1, 5), WEEKDAYS_MASK, [])
add_is_business_day("is-bday-leap-sat-2020", "synthetic-is-bday", dt.date(2020, 2, 29), WEEKDAYS_MASK, [])
add_is_business_day("is-bday-leap-thu-2024", "synthetic-is-bday", dt.date(2024, 2, 29), WEEKDAYS_MASK, [])
add_is_business_day("is-bday-sun-thu-sunday", "synthetic-is-bday", dt.date(2020, 1, 5), SUN_THU_MASK, [])
add_is_business_day("is-bday-sun-thu-friday", "synthetic-is-bday", dt.date(2020, 1, 3), SUN_THU_MASK, [])
add_is_business_day("is-bday-sun-thu-holiday", "synthetic-is-bday", dt.date(2020, 1, 5), SUN_THU_MASK, [dt.date(2020, 1, 5)])
add_is_business_day(
    "is-bday-unsorted-dup-weekend-hols",
    "synthetic-is-bday",
    dt.date(2020, 1, 3),
    WEEKDAYS_MASK,
    [dt.date(2020, 1, 4), dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
)
add_is_business_day("is-bday-min-gregorian", "synthetic-is-bday", dt.date(1, 1, 1), WEEKDAYS_MASK, [])
add_is_business_day(
    "is-bday-err-all-false-mask",
    "synthetic-errors",
    dt.date(2020, 1, 2),
    ALL_FALSE_MASK,
    [],
    rust_error="Date",
)

# 4. business_day_offset (37 cases)
# Business day with Roll::Raise
add_business_day_offset("offset-raise-zero", "synthetic-offset", dt.date(2020, 1, 2), 0, "raise", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-raise-plus1", "synthetic-offset", dt.date(2020, 1, 2), 1, "raise", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-raise-minus1-skips-hol", "synthetic-offset", dt.date(2020, 1, 2), -1, "raise", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-raise-plus5", "synthetic-offset", dt.date(2020, 1, 2), 5, "raise", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-raise-minus5", "synthetic-offset", dt.date(2020, 1, 9), -5, "raise", WEEKDAYS_MASK, NY_2020)
# Strict error cases (NumPy ValueError -> Error::Date)
add_business_day_offset("offset-err-raise-holiday", "synthetic-errors", dt.date(2020, 1, 1), 0, "raise", WEEKDAYS_MASK, NY_2020, rust_error="Date")
add_business_day_offset("offset-err-raise-weekend", "synthetic-errors", dt.date(2020, 1, 4), 1, "raise", WEEKDAYS_MASK, NY_2020, rust_error="Date")
add_business_day_offset("offset-err-all-false-mask", "synthetic-errors", dt.date(2020, 1, 2), 1, "forward", ALL_FALSE_MASK, [], rust_error="Date")
# Holiday 2020-01-01 rolling
add_business_day_offset("offset-hol-forward-0", "synthetic-offset", dt.date(2020, 1, 1), 0, "forward", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-hol-forward-1", "synthetic-offset", dt.date(2020, 1, 1), 1, "forward", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-hol-backward-0", "synthetic-offset", dt.date(2020, 1, 1), 0, "backward", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-hol-backward-m1", "synthetic-offset", dt.date(2020, 1, 1), -1, "backward", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-hol-modfol-0", "synthetic-offset", dt.date(2020, 1, 1), 0, "modified_following", WEEKDAYS_MASK, NY_2020)
add_business_day_offset("offset-hol-modprec-0", "synthetic-offset", dt.date(2020, 1, 1), 0, "modified_preceding", WEEKDAYS_MASK, NY_2020)
# Month-end holiday 2020-01-31 (Fri)
JAN31_HOL = [dt.date(2020, 1, 31)]
add_business_day_offset("offset-monthend-forward-0", "synthetic-offset", dt.date(2020, 1, 31), 0, "forward", WEEKDAYS_MASK, JAN31_HOL)
add_business_day_offset("offset-monthend-backward-0", "synthetic-offset", dt.date(2020, 1, 31), 0, "backward", WEEKDAYS_MASK, JAN31_HOL)
add_business_day_offset("offset-monthend-modfol-0", "synthetic-offset", dt.date(2020, 1, 31), 0, "modified_following", WEEKDAYS_MASK, JAN31_HOL)
add_business_day_offset("offset-monthend-modfol-plus1", "synthetic-offset", dt.date(2020, 1, 31), 1, "modified_following", WEEKDAYS_MASK, JAN31_HOL)
add_business_day_offset("offset-monthend-modfol-minus1", "synthetic-offset", dt.date(2020, 1, 31), -1, "modified_following", WEEKDAYS_MASK, JAN31_HOL)
add_business_day_offset("offset-monthend-modprec-0", "synthetic-offset", dt.date(2020, 1, 31), 0, "modified_preceding", WEEKDAYS_MASK, JAN31_HOL)
# Month-start weekend + holiday 2020-02-03 (Mon)
FEB3_HOL = [dt.date(2020, 2, 3)]
add_business_day_offset("offset-monthstart-sat-forward-0", "synthetic-offset", dt.date(2020, 2, 1), 0, "forward", WEEKDAYS_MASK, FEB3_HOL)
add_business_day_offset("offset-monthstart-sat-backward-0", "synthetic-offset", dt.date(2020, 2, 1), 0, "backward", WEEKDAYS_MASK, FEB3_HOL)
add_business_day_offset("offset-monthstart-sat-modprec-0", "synthetic-offset", dt.date(2020, 2, 1), 0, "modified_preceding", WEEKDAYS_MASK, FEB3_HOL)
add_business_day_offset("offset-monthstart-monhol-modprec-0", "synthetic-offset", dt.date(2020, 2, 3), 0, "modified_preceding", WEEKDAYS_MASK, FEB3_HOL)
# Leap day 2020-02-29 (Sat)
add_business_day_offset("offset-leap-2020-forward-0", "synthetic-offset", dt.date(2020, 2, 29), 0, "forward", WEEKDAYS_MASK, [])
add_business_day_offset("offset-leap-2020-backward-0", "synthetic-offset", dt.date(2020, 2, 29), 0, "backward", WEEKDAYS_MASK, [])
add_business_day_offset("offset-leap-2020-modfol-0", "synthetic-offset", dt.date(2020, 2, 29), 0, "modified_following", WEEKDAYS_MASK, [])
add_business_day_offset("offset-leap-2020-modprec-0", "synthetic-offset", dt.date(2020, 2, 29), 0, "modified_preceding", WEEKDAYS_MASK, [])
# All-month-holiday Feb 2020 fallback
ALL_FEB_2020 = [dt.date(2020, 2, d) for d in range(1, 30)]
add_business_day_offset("offset-allfeb-modfol-0", "synthetic-offset", dt.date(2020, 2, 15), 0, "modified_following", WEEKDAYS_MASK, ALL_FEB_2020)
add_business_day_offset("offset-allfeb-modfol-plus1", "synthetic-offset", dt.date(2020, 2, 15), 1, "modified_following", WEEKDAYS_MASK, ALL_FEB_2020)
add_business_day_offset("offset-allfeb-modprec-0", "synthetic-offset", dt.date(2020, 2, 15), 0, "modified_preceding", WEEKDAYS_MASK, ALL_FEB_2020)
add_business_day_offset("offset-allfeb-modprec-minus1", "synthetic-offset", dt.date(2020, 2, 15), -1, "modified_preceding", WEEKDAYS_MASK, ALL_FEB_2020)
# Custom weekmasks & large offsets
add_business_day_offset("offset-sun-thu-fri-forward-2", "synthetic-offset", dt.date(2020, 1, 3), 2, "forward", SUN_THU_MASK, [dt.date(2020, 1, 5)])
add_business_day_offset("offset-large-plus-5000", "synthetic-offset", dt.date(2000, 1, 3), 5000, "raise", WEEKDAYS_MASK, [dt.date(2000, 1, 1), dt.date(2010, 7, 5), dt.date(2015, 12, 25)])
add_business_day_offset("offset-large-minus-5000", "synthetic-offset", dt.date(2020, 1, 2), -5000, "raise", WEEKDAYS_MASK, [dt.date(2000, 1, 1), dt.date(2010, 7, 5), dt.date(2015, 12, 25)])
# Documented Gregorian 0001..=9999 bounds (NumPy datetime64[D] allows year 0 / year 10000; Rust returns Error::Date)
add_business_day_offset(
    "offset-doc-underflow-0001",
    "documented-parity-boundaries",
    dt.date(1, 1, 1),
    -1,
    "backward",
    WEEKDAYS_MASK,
    [],
    documented_diff="gregorian_date_range_bounds",
    expected_rust_error="Date",
)
add_business_day_offset(
    "offset-doc-overflow-9999",
    "documented-parity-boundaries",
    dt.date(9999, 12, 31),
    1,
    "forward",
    WEEKDAYS_MASK,
    [],
    documented_diff="gregorian_date_range_bounds",
    expected_rust_error="Date",
)

# 5. business_day_count (16 cases)
add_business_day_count("count-asym-hol-forward", "synthetic-count", dt.date(2020, 1, 1), dt.date(2020, 1, 6), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-asym-hol-reverse", "synthetic-count", dt.date(2020, 1, 6), dt.date(2020, 1, 1), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-asym-weekend-forward", "synthetic-count", dt.date(2020, 1, 4), dt.date(2020, 1, 6), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-asym-weekend-reverse", "synthetic-count", dt.date(2020, 1, 6), dt.date(2020, 1, 4), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-bday-forward", "synthetic-count", dt.date(2020, 1, 2), dt.date(2020, 1, 6), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-bday-reverse", "synthetic-count", dt.date(2020, 1, 6), dt.date(2020, 1, 2), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-equal-bday", "synthetic-count", dt.date(2020, 1, 6), dt.date(2020, 1, 6), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-equal-hol", "synthetic-count", dt.date(2020, 1, 1), dt.date(2020, 1, 1), WEEKDAYS_MASK, NY_2020)
add_business_day_count("count-leap-2020-sat", "synthetic-count", dt.date(2020, 2, 24), dt.date(2020, 3, 2), WEEKDAYS_MASK, [])
add_business_day_count("count-leap-2024-thu", "synthetic-count", dt.date(2024, 2, 26), dt.date(2024, 3, 4), WEEKDAYS_MASK, [])
DECADE_HOLS = [dt.date(1980, 1, 1), dt.date(2000, 2, 29), dt.date(2020, 1, 1), dt.date(2024, 2, 29), dt.date(2040, 7, 4)]
add_business_day_count("count-multi-decade-forward", "synthetic-count", dt.date(1970, 1, 1), dt.date(2050, 1, 1), WEEKDAYS_MASK, DECADE_HOLS)
add_business_day_count("count-multi-decade-reverse", "synthetic-count", dt.date(2050, 1, 1), dt.date(1970, 1, 1), WEEKDAYS_MASK, DECADE_HOLS)
add_business_day_count("count-sun-thu-custom", "synthetic-count", dt.date(2020, 1, 1), dt.date(2020, 1, 15), SUN_THU_MASK, [dt.date(2020, 1, 5)])
add_business_day_count("count-sat-only-custom", "synthetic-count", dt.date(2020, 1, 1), dt.date(2020, 2, 1), SAT_ONLY_MASK, [dt.date(2020, 1, 11)])
add_business_day_count("count-all-days-custom", "synthetic-count", dt.date(2020, 2, 1), dt.date(2020, 3, 1), ALL_DAYS_MASK, [dt.date(2020, 2, 29)])
add_business_day_count("count-err-all-false-mask", "synthetic-errors", dt.date(2020, 1, 1), dt.date(2020, 1, 6), ALL_FALSE_MASK, [], rust_error="Date")

payload = {
    "schema": "gsquant-calendar-parity-v1",
    "upstream_sha": "70220e300524d54bd273be0f4c485d53b3780482",
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "python_lock_sha256": hashlib.sha256((ROOT / "python-lock.txt").read_bytes()).hexdigest(),
    "source": [
        "gs_quant/datetime/date.py:has_feb_29",
        "gs_quant/datetime/date.py:day_count_fraction",
        "gs_quant/datetime/date.py:is_business_day",
        "gs_quant/datetime/date.py:business_day_offset",
        "gs_quant/datetime/date.py:business_day_count",
        "numpy:busdaycalendar",
    ],
    "cases": cases,
}
raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
(ROOT / "calendar-vectors.json").write_bytes(raw)
print(f"generated {len(cases)} upstream calendar cases")
