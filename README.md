비공식 Rust 포팅 — 원본 goldmansachs/gs-quant, Goldman Sachs 와 무관

Unofficial Rust port of goldmansachs/gs-quant; not affiliated with, endorsed or sponsored by Goldman Sachs.

Quantus Analytics is a unified repository for financial calculations. Its first package, `quantus-math`, provides offline, dependency-free moving averages, returns, realized volatility, and correlation as a slice of [gs-quant](https://github.com/goldmansachs/gs-quant), based on upstream commit `70220e300524d54bd273be0f4c485d53b3780482`. Prices and absolute returns use the caller's price units; simple returns, logarithmic returns, and correlation are dimensionless ratios (not percent); volatility is returned in percent (e.g. `20.0` for 20%). No network, market data, or internal engine is involved.

| Feature | Supported subset |
|---|---|
| Index | Strictly increasing unique Gregorian dates, years 1–9999; no times or time zones. Length mismatch and duplicate or unsorted dates are errors. |
| Moving average | Expanding default (`Window::Expanding`), observation window with observation ramp, or calendar-day window with calendar-day ramp. Days use `(date - size, date]`; ramp keeps dates at or after first date + ramp. NaN values are ignored; a window with no non-NaN values gives NaN. |
| Returns | Positive integer observation lag, SIMPLE, LOGARITHMIC and ABSOLUTE. NaN propagates. Logarithmic returns reject nonpositive prices. Division by zero follows floating-point arithmetic. |
| Volatility | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<ReturnKind>` (`Simple`, `Logarithmic`, `Absolute`, or `None` for precomputed returns). Standard sample volatility (`assume_zero_mean = false`, `n - 1` denominator, `NaN` for `< 2` non-NaN values) and zero-mean RMS (`assume_zero_mean = true`, `n` denominator, `NaN` for `0` non-NaN values). Explicit `Option<u32>` annualization factor (including `0`, which scales finite values to `0.0` and keeps `NaN`), or `None` for upstream automatic classification (`< 2.1d → 252`, `[6, 8) → 52`, `[14, 17) → 26`, `[25, 35) → 12`, `[85, 97) → 4`, `[360, 386) → 1`). Output is annualized in percent (`* sqrt(factor) * 100`). |
| Correlation | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<(ReturnKind, ReturnKind)>` (`Some((kx, ky))` computes lag-1 returns on `x` and `y` independently before date intersection; `None` treats both inputs as precomputed returns). Aligns common dates and drops any pair with a `NaN` before rolling (`Observations` counts aligned non-NaN pairs; `Days` uses `(date - size, date]`). Reindexes results onto `x`'s dates (unmatched left dates become `NaN`) and applies the outer ramp on `x`. Returns a dimensionless ratio (not percent). Sample Pearson (`assume_zero_mean = false`) yields `NaN` when a window has `< 2` pairs or zero variance. Zero-mean mode (`assume_zero_mean = true`) computes normalized inner products via upstream's `mean` + zero-mean `volatility` pipeline on the aligned pairs: both the inner window/ramp (on aligned pairs) and outer ramp (on `x`) apply, so dates masked by the inner ramp remain `NaN` even if they survive the outer ramp; empty, singleton, irregular-spacing, or oversize-window aligned pairs (including `Window::Expanding` when aligned count `< x.len()`) return `Err(Error::AnnualizationFactor)`, and inner observation ramp `> aligned.len()` returns `Err(Error::Ramp)`. |
| Numerical precision & non-finite inputs | Parity tolerance `1e-12 + 1e-10 * |expected|` applies to well-conditioned finite inputs. Inputs with `±Inf` are excluded from compatibility vectors: Rust follows IEEE-754 arithmetic (`NaN` for sample volatility and Pearson correlation, `+Inf` for RMS, `NaN` for zero-mean `Inf / Inf`), whereas upstream pandas `rolling()` and numpy `rolling_std()` diverge on `±Inf`. For constant non-power-of-two float series (e.g. `0.01`, `0.1`), Rust detects finite constant windows before centering and returns `NaN` (zero variance), whereas upstream pandas 2.3.3 observation-window `rolling().corr()` can emit `±Inf` from covariance rounding residue divided by zero standard deviation (e.g. on `0.01`, or `0.1` vs varying), and upstream calendar-day `Series.corr` (`np.corrcoef`) can emit rounding-residue `1.0` (constant/constant) or `0.0` (constant/varying) when `sum(x) / n != x` (e.g. three `0.1` values). For inputs with very large constant offsets relative to spread (e.g. `1e9 + 0.1`, where `f64` input ULP is `~1.2e-7`), Rust uses two-pass centered sums (`sum((x - mean)^2)` / `sum((x - mean_x)(y - mean_y))`), avoiding catastrophic cancellation while upstream pandas observation-window and numpy day-window paths diverge by `~6e-7` relative error (outside the `1e-10` parity bound). |
| Not supported | Date-based return lag, months/weeks (`1m`, `3m`, `1w`), mixed-unit ramps, duplicate or unsorted indices, time zones, pandas nullable values distinct from NaN, volatility/variance/correlation swap wrappers (`vol_swap_volatility`, `corr_swap_correlation`), and other gs-quant functions. This is not a claim of full upstream compatibility. |

Observation window size must be positive and its ramp at most the number of rows; for `moving_average` and Pearson `correlation`, size larger than `x` returns an empty series. For `volatility`, size larger than the series returns empty when `annualization_factor` is `Some(...)`, and `Err(Error::AnnualizationFactor)` when `None`. For zero-mean `correlation`, size larger than the aligned non-NaN pair count (including `Window::Expanding` when `aligned.len() < x.len()`) returns `Err(Error::AnnualizationFactor)` unless an inner observation ramp `> aligned.len()` triggers `Err(Error::Ramp)` first. Calendar-day size must be positive. Empty `moving_average`, `volatility`, and left-input `correlation` return empty before window or annualization validation; `returns` rejects lag zero even on empty input. A larger-than-series lag gives all NaNs. Singleton or unclassifiable spacing under automatic annualization returns `Err(Error::AnnualizationFactor)`. Use `Window::Observations { size: n, ramp: n }` for the upstream integer-window default ramp and `Window::Days { size: n, ramp: n }` for its day-string default. Public errors are `Error` variants.

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

Standalone offline `correlation` example for an external Cargo project:

```rust
use quantus_math::{correlation, Date, Error, ReturnKind, Series, Window};

fn main() {
    let d = |m, day| Date::new(2024, m, day).unwrap();
    // Two close-price series on different trading calendars (Mar 2 vs Mar 4)
    let x = Series::try_new(
        vec![d(3, 1), d(3, 2), d(3, 3), d(3, 5), d(3, 6)],
        vec![10.0, 12.0, 14.0, 13.0, 16.0],
    )
    .unwrap();
    let y = Series::try_new(
        vec![d(3, 1), d(3, 3), d(3, 4), d(3, 5), d(3, 6)],
        vec![20.0, 24.0, 22.0, 19.0, 25.0],
    )
    .unwrap();

    // 1. Default expanding window on mismatched dates (unmatched Mar 2 is NaN)
    let def = correlation(
        &x,
        &y,
        Window::Expanding,
        Some((ReturnKind::Simple, ReturnKind::Simple)),
        false,
    )
    .unwrap();
    assert_eq!(def.dates(), x.dates());
    assert!(def.values()[0].is_nan() && def.values()[1].is_nan() && def.values()[2].is_nan());
    assert!((def.values()[4] - 0.9989243898423678).abs() < 1e-10);

    // 2. Observation window vs calendar-day window with mixed return kinds (log, absolute)
    let mixed = Some((ReturnKind::Logarithmic, ReturnKind::Absolute));
    let obs = correlation(&x, &y, Window::Observations { size: 3, ramp: 0 }, mixed, false).unwrap();
    let cal = correlation(&x, &y, Window::Days { size: 3, ramp: 2 }, mixed, false).unwrap();
    assert!((obs.values()[4] - 0.9994382571107351).abs() < 1e-10);
    assert_eq!(cal.dates(), &[d(3, 3), d(3, 5), d(3, 6)]);
    assert!((cal.values()[2] - 1.0).abs() < 1e-10);

    // 3. Precomputed returns: zero-mean vs Pearson, plus irregular-spacing error and empty left
    let r1 = Series::try_new(
        vec![d(2, 26), d(2, 28), d(2, 29), d(3, 1), d(3, 4), d(3, 6)],
        vec![0.01, 0.03, 0.02, 0.02, 0.01, 0.04],
    )
    .unwrap();
    let r2 = Series::try_new(r1.dates().to_vec(), vec![0.02, -0.02, 0.04, 0.01, 0.03, 0.01])
        .unwrap();
    let zm = correlation(&r1, &r2, Window::Observations { size: 2, ramp: 0 }, None, true).unwrap();
    assert!((zm.values()[5] - 0.5368754921931593).abs() < 1e-10);

    let irregular = Series::try_new(
        vec![d(2, 26), d(2, 29), d(3, 3)],
        vec![0.01, 0.03, 0.02],
    )
    .unwrap();
    assert_eq!(
        correlation(&irregular, &irregular, Window::Observations { size: 2, ramp: 0 }, None, true),
        Err(Error::AnnualizationFactor)
    );
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(correlation(&empty, &y, Window::Expanding, None, true)
        .unwrap()
        .values()
        .is_empty());

    println!("expanding={} obs={} cal={} zm={}", def.values()[4], obs.values()[4], cal.values()[2], zm.values()[5]);
}
```

Parity: `python3 parity/check.py --replay`, `python3 parity/check-volatility.py --replay`, and `python3 parity/check-correlation.py --replay` build the offline replay binary and check frozen unrounded Python outputs. [The assertion map](parity/MAP.md) records the covered cases and exclusions. To regenerate vectors, use the pinned Python environment and an independently verified checkout of the upstream SHA:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-volatility.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-correlation.py /path/to/gs-quant
```

Generation imports the upstream Python implementation; replay does not. Compare a regenerated file and its hash before accepting any change. Upstream LICENSE, NOTICE and NOTICE.txt are preserved; source attribution is in `src/lib.rs`.
