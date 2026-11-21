비공식 Rust 포팅 — 원본 goldmansachs/gs-quant, Goldman Sachs 와 무관

Unofficial Rust port of goldmansachs/gs-quant; not affiliated with, endorsed or sponsored by Goldman Sachs.

Quantus Analytics is a unified repository for financial calculations. Its first package, `quantus-math`, provides offline, dependency-free moving averages, returns, and realized volatility as a slice of [gs-quant](https://github.com/goldmansachs/gs-quant), based on upstream commit `70220e300524d54bd273be0f4c485d53b3780482`. Prices and absolute returns use the caller's price units; simple and logarithmic returns are dimensionless ratios; volatility is returned in percent (e.g. `20.0` for 20%). No network, market data, or internal engine is involved.

| Feature | Supported subset |
|---|---|
| Index | Strictly increasing unique Gregorian dates, years 1–9999; no times or time zones. Length mismatch and duplicate or unsorted dates are errors. |
| Moving average | Expanding default (`Window::Expanding`), observation window with observation ramp, or calendar-day window with calendar-day ramp. Days use `(date - size, date]`; ramp keeps dates at or after first date + ramp. NaN values are ignored; a window with no non-NaN values gives NaN. |
| Returns | Positive integer observation lag, SIMPLE, LOGARITHMIC and ABSOLUTE. NaN propagates. Logarithmic returns reject nonpositive prices. Division by zero follows floating-point arithmetic. |
| Volatility | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<ReturnKind>` (`Simple`, `Logarithmic`, `Absolute`, or `None` for precomputed returns). Standard sample volatility (`assume_zero_mean = false`, `n - 1` denominator, `NaN` for `< 2` non-NaN values) and zero-mean RMS (`assume_zero_mean = true`, `n` denominator, `NaN` for `0` non-NaN values). Explicit `Option<u32>` annualization factor (including `0`, which scales finite values to `0.0` and keeps `NaN`), or `None` for upstream automatic classification (`< 2.1d → 252`, `[6, 8) → 52`, `[14, 17) → 26`, `[25, 35) → 12`, `[85, 97) → 4`, `[360, 386) → 1`). Output is annualized in percent (`* sqrt(factor) * 100`). |
| Numerical precision & non-finite inputs | Parity tolerance `1e-12 + 1e-10 * |expected|` applies to well-conditioned finite inputs. Inputs with `±Inf` are excluded from compatibility vectors: Rust follows IEEE-754 arithmetic (`NaN` for sample volatility, `+Inf` for RMS), whereas upstream pandas `rolling()` and numpy `rolling_std()` diverge on `±Inf`. For inputs with very large constant offsets relative to spread (e.g. `1e9 + 0.1`, where `f64` input ULP is `~1.2e-7`), Rust uses two-pass centered variance (`sum((x - mean)^2) / (n - 1)`), avoiding catastrophic cancellation and tracking numpy `rolling_std`, while upstream pandas observation-window and numpy day-window paths diverge by `~6e-7` relative error (outside the `1e-10` parity bound). |
| Not supported | Date-based return lag, months/weeks (`1m`, `3m`, `1w`), mixed-unit ramps, duplicate indices, time zones, pandas nullable values distinct from NaN, correlation, volatility/variance swaps, and other gs-quant functions. This is not a claim of full upstream compatibility. |

Observation window size must be positive and its ramp at most the number of rows; for `moving_average`, size larger than the series returns an empty series. For `volatility`, size larger than the series returns empty when `annualization_factor` is `Some(...)`, and `Err(Error::AnnualizationFactor)` when `None`. Calendar-day size must be positive. Empty `moving_average` and `volatility` inputs return empty before window or annualization validation; `returns` rejects lag zero even on empty input. A larger-than-series lag gives all NaNs. Singleton or unclassifiable spacing under automatic annualization returns `Err(Error::AnnualizationFactor)`. Use `Window::Observations { size: n, ramp: n }` for the upstream integer-window default ramp and `Window::Days { size: n, ramp: n }` for its day-string default. Public errors are `Error` variants.

Run the independent example twice without network:

```sh
cargo run --offline --locked --example offline
cargo run --offline --locked --example offline
```

To use in an external Cargo project depending on `quantus-math`:

```rust
use quantus_math::{moving_average, returns, volatility, Date, Error, ReturnKind, Series, Window};

fn main() {
    let dates: Vec<_> = (1..=6).map(|d| Date::new(2019, 1, d).unwrap()).collect();
    let prices = Series::try_new(
        dates,
        vec![100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792],
    )
    .unwrap();

    let mean = moving_average(&prices, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(mean.values()[1], 100.5);
    let simple = returns(&prices, 1, ReturnKind::Simple).unwrap();
    assert!((simple.values()[1] - 0.01).abs() < 1e-12);

    // 1. Annualized realized volatility (%) from close prices
    let vol_pct = volatility(
        &prices,
        Window::Expanding,
        Some(ReturnKind::Simple),
        None,
        false,
    )
    .unwrap();
    assert!((vol_pct.values()[5] - 26.563132345414413).abs() < 1e-10);

    // 2. Precomputed returns with NaN: 3-observation vs 3-calendar-day vs zero-mean RMS
    let rets = Series::try_new(
        vec![
            Date::new(2024, 2, 26).unwrap(),
            Date::new(2024, 2, 28).unwrap(),
            Date::new(2024, 2, 29).unwrap(),
            Date::new(2024, 3, 1).unwrap(),
            Date::new(2024, 3, 4).unwrap(),
            Date::new(2024, 3, 6).unwrap(),
        ],
        vec![0.01, 0.03, f64::NAN, 0.02, 0.01, 0.04],
    )
    .unwrap();
    let vol_obs =
        volatility(&rets, Window::Observations { size: 3, ramp: 0 }, None, None, false).unwrap();
    let vol_3d = volatility(&rets, Window::Days { size: 3, ramp: 0 }, None, None, false).unwrap();
    let vol_rms =
        volatility(&rets, Window::Observations { size: 3, ramp: 0 }, None, None, true).unwrap();
    assert!((vol_obs.values()[5] - 24.24871130596428).abs() < 1e-10);
    assert!((vol_3d.values()[5] - 33.67491648096548).abs() < 1e-10);
    assert!((vol_rms.values()[5] - 42.0).abs() < 1e-10);

    // 3. Irregular spacing error under auto annualization, and empty input
    let irregular = Series::try_new(
        vec![
            Date::new(2024, 2, 26).unwrap(),
            Date::new(2024, 2, 29).unwrap(),
            Date::new(2024, 3, 3).unwrap(),
        ],
        vec![0.01, 0.03, 0.02],
    )
    .unwrap();
    assert_eq!(
        volatility(&irregular, Window::Expanding, None, None, false),
        Err(Error::AnnualizationFactor)
    );
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(volatility(&empty, Window::Expanding, Some(ReturnKind::Simple), None, false)
        .unwrap()
        .values()
        .is_empty());
}
```

Parity: `python3 parity/check.py --replay` and `python3 parity/check-volatility.py --replay` build the offline replay binary and check frozen unrounded Python outputs. [The assertion map](parity/MAP.md) records the covered cases and exclusions. To regenerate vectors, use the pinned Python environment and an independently verified checkout of the upstream SHA:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-volatility.py /path/to/gs-quant
```

Generation imports the upstream Python implementation; replay does not. Compare a regenerated file and its hash before accepting any change. Upstream LICENSE, NOTICE and NOTICE.txt are preserved; source attribution is in `src/lib.rs`.
