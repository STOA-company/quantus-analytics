비공식 Rust 포팅 — 원본 goldmansachs/gs-quant, Goldman Sachs 와 무관

Unofficial Rust port of goldmansachs/gs-quant; not affiliated with, endorsed or sponsored by Goldman Sachs.

Quantus Analytics is a unified repository for financial calculations. Its first package, `quantus-math`, provides offline, dependency-free moving averages and returns as a slice of [gs-quant](https://github.com/goldmansachs/gs-quant), based on upstream commit `70220e300524d54bd273be0f4c485d53b3780482`. Prices and absolute returns use the caller's price units; simple and logarithmic returns are dimensionless ratios, not percentages. No network, market data, or internal engine is involved.

| Feature | Supported subset |
|---|---|
| Index | Strictly increasing unique Gregorian dates, years 1–9999; no times or time zones. Length mismatch and duplicate or unsorted dates are errors. |
| Moving average | Expanding default (`Window::Expanding`), observation window with observation ramp, or calendar-day window with calendar-day ramp. Days use `(date - size, date]`; ramp keeps dates at or after first date + ramp. NaN values are ignored; a window with no non-NaN values gives NaN. |
| Returns | Positive integer observation lag, SIMPLE, LOGARITHMIC and ABSOLUTE. NaN propagates. Logarithmic returns reject nonpositive prices. Division by zero follows floating-point arithmetic. |
| Not supported | Date-based return lag, months/weeks, mixed-unit ramps, duplicate indices, time zones, pandas nullable values distinct from NaN, and other gs-quant functions. This is not a claim of full upstream compatibility. |

Observation window size must be positive and its ramp at most the number of rows; size larger than the series returns an empty series. Calendar-day size must be positive. Empty moving-average input returns empty before window validation; returns rejects lag zero even on empty input. A larger-than-series lag gives all NaNs. Use `Window::Observations { size: n, ramp: n }` for the upstream integer-window default ramp and `Window::Days { size: n, ramp: n }` for its day-string default. Public errors are `Error` variants. Inputs with infinity are not covered by the compatibility vectors.

Run the independent example twice without network:

```sh
cargo run --offline --locked --example offline
cargo run --offline --locked --example offline
```

To use in Rust:

```rust
use quantus_math::{moving_average, returns, Date, ReturnKind, Series, Window};
let dates = vec![Date::new(2019, 1, 1).unwrap(), Date::new(2019, 1, 2).unwrap()];
let prices = Series::try_new(dates, vec![3.0, 2.0]).unwrap();
let mean = moving_average(&prices, Window::Observations { size: 2, ramp: 0 }).unwrap();
assert_eq!(mean.values(), &[3.0, 2.5]);
let change = returns(&prices, 1, ReturnKind::Simple).unwrap();
assert!((change.values()[1] - (-1.0 / 3.0)).abs() < 1e-12);
```

Parity: `python3 parity/check.py --replay` builds the offline example and checks frozen unrounded Python outputs. [The assertion map](parity/MAP.md) records the covered cases and exclusions. To regenerate vectors, use the pinned Python environment and an independently verified checkout of the upstream SHA:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate.py /path/to/gs-quant
```

Generation imports the upstream Python implementation; replay does not. Compare a regenerated file and its hash before accepting any change. Upstream LICENSE, NOTICE and NOTICE.txt are preserved; source attribution is in `src/lib.rs`.
