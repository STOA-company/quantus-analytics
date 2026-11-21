# Upstream assertion map

Source SHA: `70220e300524d54bd273be0f4c485d53b3780482`.
`gs_quant/test/timeseries/test_technicals.py::test_moving_average`: default → `ma_default`; Window(1,0) → `ma_1`; Window(2,0) → `ma_2`; '2d' → `ma_2d`; empty → `ma_empty`.

`gs_quant/test/timeseries/test_econometrics.py::test_returns`: empty → `returns_empty`; default → `returns_default`; SIMPLE 1 and 2 → `returns_simple_1`, `returns_simple_2`; LOGARITHMIC 1 and 2 → `returns_logarithmic_1`, `returns_logarithmic_2`; ABSOLUTE 1 and 2 → `returns_absolute_1`, `returns_absolute_2`. Date lag '2d' is omitted: upstream returns a DatetimeIndex, which this date-only API does not support. Invalid return kind is excluded from numeric replay because the Rust enum makes it unrepresentable. The upstream pytest exercises both omitted branches. Additional vectors `ma_nan` and `returns_nan` cover NaN positions.

`check.py --replay` compares all retained cases, every date, NaN/Inf classification and unrounded finite value with tolerance `1e-12 + 1e-10 * abs(expected)`. Both hashes and vectors are generated from Python only. `tests/contracts.rs` verifies this subset's validation and leap-day behavior independently; validation is deliberately narrower than upstream in some cases. The upstream `quant_extensions` optional rolling implementation is not used by these moving-average or returns paths.

For empty upstream results the vector records the actual pandas `RangeIndex`; Rust has a typed empty date index, so replay verifies the empty shape and maps that index deliberately rather than claiming identical index types. Nonempty date indices compare each date exactly. The full upstream tree identifies Scott Weinstein in `dco/Scott Weinstein.dco`; no attribution to the translated files was found in their available Git history. Both original NOTICE files are preserved and the source header includes the upstream notice.
