비공식 Rust 포팅 — 원본 goldmansachs/gs-quant, Goldman Sachs 와 무관

Unofficial Rust port of goldmansachs/gs-quant; not affiliated with, endorsed or sponsored by Goldman Sachs.

Quantus Analytics is a unified repository for financial calculations. Its first package, `quantus-math`, provides offline, dependency-free moving averages, exponential moving averages, returns, realized volatility, correlation, beta, and maximum drawdown as a slice of [gs-quant](https://github.com/goldmansachs/gs-quant), based on upstream commit `70220e300524d54bd273be0f4c485d53b3780482`. Prices, moving averages, exponential moving averages, and absolute returns use the caller's price units; simple returns, logarithmic returns, correlation, beta, and maximum drawdown are dimensionless ratios (not percent; e.g. `-0.2` for a 20% drawdown); volatility is returned in percent (e.g. `20.0` for 20%). No network, market data, or internal engine is involved.

| Feature | Supported subset |
|---|---|
| Index | Strictly increasing unique Gregorian dates, years 1–9999; no times or time zones. Length mismatch and duplicate or unsorted dates are errors. |
| Moving average | Expanding default (`Window::Expanding`), observation window with observation ramp, or calendar-day window with calendar-day ramp. Days use `(date - size, date]`; ramp keeps dates at or after first date + ramp. NaN values are ignored; a window with no non-NaN values gives NaN. |
| Exponential moving average | Reuses `Series`, `Date`, and `Error::Beta`. `exponential_moving_average(x, beta: Option<f64>)` uses upstream default `beta = 0.75` when `None`, and decays over observation (row) order rather than calendar-day spacing, matching upstream `x.ewm(alpha=1 - beta, adjust=False).mean()`. Leading `NaN` and `±Inf` values emit `NaN` until the first finite observation; subsequent `NaN` and `±Inf` values are treated as missing, carrying forward the prior average while decaying its weight across missing rows when the next finite observation arrives. Equal finite values preserve the prior value and sign (including `-0.0 == +0.0`, even at `beta = 0.0`). Valid weights satisfy `0.0 <= beta < 1.0` (`-0.0` is accepted as `0.0`); negative values, values `>= 1.0`, `±Inf`, and `NaN` return `Err(Error::Beta)` even on empty input. |
| Returns | Positive integer observation lag, SIMPLE, LOGARITHMIC and ABSOLUTE. NaN propagates. Logarithmic returns reject nonpositive prices. Division by zero follows floating-point arithmetic. |
| Volatility | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<ReturnKind>` (`Simple`, `Logarithmic`, `Absolute`, or `None` for precomputed returns). Standard sample volatility (`assume_zero_mean = false`, `n - 1` denominator, `NaN` for `< 2` non-NaN values) and zero-mean RMS (`assume_zero_mean = true`, `n` denominator, `NaN` for `0` non-NaN values). Explicit `Option<u32>` annualization factor (including `0`, which scales finite values to `0.0` and keeps `NaN`), or `None` for upstream automatic classification (`< 2.1d → 252`, `[6, 8) → 52`, `[14, 17) → 26`, `[25, 35) → 12`, `[85, 97) → 4`, `[360, 386) → 1`). Output is annualized in percent (`* sqrt(factor) * 100`). |
| Correlation | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<(ReturnKind, ReturnKind)>` (`Some((kx, ky))` computes lag-1 returns on `x` and `y` independently before date intersection; `None` treats both inputs as precomputed returns). Aligns common dates and drops any pair with a `NaN` before rolling (`Observations` counts aligned non-NaN pairs; `Days` uses `(date - size, date]`). Reindexes results onto `x`'s dates (unmatched left dates become `NaN`) and applies the outer ramp on `x`. Returns a dimensionless ratio (not percent). Sample Pearson (`assume_zero_mean = false`) yields `NaN` when a window has `< 2` pairs or zero variance. Zero-mean mode (`assume_zero_mean = true`) computes normalized inner products via upstream's `mean` + zero-mean `volatility` pipeline on the aligned pairs: both the inner window/ramp (on aligned pairs) and outer ramp (on `x`) apply, so dates masked by the inner ramp remain `NaN` even if they survive the outer ramp; empty, singleton, irregular-spacing, or oversize-window aligned pairs (including `Window::Expanding` when aligned count `< x.len()`) return `Err(Error::AnnualizationFactor)`, and inner observation ramp `> aligned.len()` returns `Err(Error::Ramp)`. |
| Beta | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `ReturnKind::Simple` (internally when `prices = true`), `Error`, and outer ramp. When `prices = true`, computes lag-1 simple returns independently on `x` and `benchmark`; when `prices = false`, treats both series as precomputed returns. For `Window::Expanding` (size = `x.dates().len()`) and `Window::Observations`, computes rolling sample covariance (`ddof = 1`, skipping `NaN` pairs) over the outer-joined date union divided by `benchmark`'s own rolling sample variance (`ddof = 1`, skipping `NaN`s) aligned by date—without pooling the two sample sets. For `Window::Days`, computes sample covariance divided by benchmark variance over `(date - size, date]` on the date intersection, propagating any `NaN` inside the window. Masks the first 3 positions of the intermediate result (`union` for observations/expanding, `intersection` for days) to `NaN`, reindexes onto `x`'s dates (`NaN` for unmatched dates), and applies the outer ramp. An exactly constant finite benchmark window yields `NaN` (zero denominator). |
| Max drawdown | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `Date`, ramp, and `Error`. Two-stage rolling peak-to-trough ratio: computes the rolling maximum price, forms `price / rolling_max - 1`, and takes the rolling minimum of those ratios over the same window before applying the ramp (e.g. `-0.2` for a 20% drawdown). `Window::Expanding` uses `x.dates().len()` with ramp `0`; `Window::Observations` skips `NaN` and `±Inf` in both rolling stages (matching upstream pandas `rolling().max()` / `rolling().min()`); `Window::Days` uses `(date - size, date]` in both stages without interpolating holidays, skipping `NaN` while retaining `±Inf` (matching upstream pandas `Series.max()` / `Series.min()`). All-NaN windows and `0 / 0` stay `NaN`. Empty input returns empty for `Expanding`, `Observations`, and `Days { ramp: 0 }`, and returns `Err(Error::Ramp)` for `Days { ramp > 0 }` (mapping upstream's `IndexError` when applying a positive calendar ramp to an empty date index). For `Window::Days`, date parity with Python holds only when the input dates, every left window boundary (`date - size` days), and the ramp cutoff (`first_date + ramp` days) are representable and computable with the pinned pandas `DateOffset`. Rust evaluates calendar windows and ramps via integer day-count differences between `Date` ordinals (without constructing pandas timestamps or `DateOffset` values), so extremes that raise Python date-range or `DateOffset` errors (`OverflowError` / `OutOfBoundsDatetime`) can still produce normal values or an empty `Series` in Rust—for example, on `2024-01-01`/`2024-01-05` with prices `[2.0, 1.0]`, an ordinary window such as `size: 100, ramp: 0` matches Python (`[0.0, -0.5]`), whereas `size: 1_000_000` returns `[0.0, -0.5]` and `ramp: 1_000_000` returns empty in Rust while Python errors; similarly, near pandas's lower timestamp limit, `1677-09-22`/`1677-09-23` with `size: 2` succeeds in Rust (`[0.0, -0.5]`) even though the left boundary `date - 2` days is out of range for pandas. |
| Numerical precision & non-finite inputs | Parity tolerance `1e-12 + 1e-10 * |expected|` applies to well-conditioned finite inputs. For `volatility`, `correlation`, and `beta`, inputs with `±Inf` are excluded from compatibility vectors: Rust follows IEEE-754 arithmetic (`NaN` for sample volatility, Pearson correlation, and beta, `+Inf` for RMS, `NaN` for zero-mean `Inf / Inf`), whereas upstream pandas `rolling()` and numpy `rolling_std()` diverge on `±Inf`. For `max_drawdown`, `±Inf` (including `-Inf` produced by a negative price following a `+0.0` peak) matches upstream: integer/expanding windows skip `±Inf` in both rolling stages while calendar-day windows retain `±Inf`, except when a calendar-day window's rolling peak ties across mixed-sign zeros (`+0.0` and `-0.0`). On mixed-sign zero peak ties in `Window::Days`, Rust keeps the last zero in the window (updating on `!(cur > v)`), so a negative price divides by that zero's sign (`-Inf` when the last zero is `+0.0`, `+Inf` when it is `-0.0`); even within the same pinned Python/NumPy/pandas environment, the resulting `±Inf` sign for mixed-sign zeros can vary depending on NumPy CPU optimizations. For constant non-power-of-two float series (e.g. `0.01`, `0.1`), Rust detects finite constant windows before centering and returns `NaN` (zero variance / zero denominator), whereas upstream pandas 2.3.3 observation-window `rolling().corr()` can emit `±Inf` from covariance rounding residue divided by zero standard deviation (e.g. on `0.01`, or `0.1` vs varying), upstream calendar-day `Series.corr` (`np.corrcoef`) can emit rounding-residue `1.0` (constant/constant) or `0.0` (constant/varying) when `sum(x) / n != x` (e.g. three `0.1` values), and upstream calendar-day `beta` (`np.cov` / `np.var`) emits `±21.333333333333332` (`±64/3`) on constant `0.1` over a 6-day window (`constant-calendar` vector) due to floating-point rounding residue in the denominator variance. For inputs with very large constant offsets relative to spread (e.g. `1e9 + 0.1`, where `f64` input ULP is `~1.2e-7`), Rust uses two-pass centered sums (`sum((x - mean)^2)` / `sum((x - mean_x)(y - mean_y))`), avoiding catastrophic cancellation while upstream pandas observation-window and numpy day-window paths diverge by `~6e-7` relative error (outside the `1e-10` parity bound). For `exponential_moving_average`, `±Inf` inputs match upstream pandas (treated as missing like `NaN`), and signed zeros match upstream bit-for-bit; for `beta = NaN`, upstream `gs-quant` documents `0 <= beta < 1` (`technicals.py:265-266`) while pandas `get_center_of_mass` checks `alpha <= 0 or alpha > 1` and lets `NaN` slip through (producing alternating artifacts such as `[2.0, NaN, 4.0]` on `[2.0, 3.0, 4.0]`), whereas Rust strictly enforces `[0.0, 1.0)` and returns `Err(Error::Beta)`. |
| Not supported | Date-based return lag, months/weeks (`1m`, `3m`, `1w`), mixed-unit ramps, duplicate or unsorted indices, `DatetimeIndex` / time zones, non-date indices (`TypeError` in upstream `max_drawdown`), pandas nullable values distinct from NaN, non-simple return kinds for `beta`, volatility/variance/correlation swap wrappers (`vol_swap_volatility`, `corr_swap_correlation`), service-level portfolio beta (`measures_portfolios.py`) and drawdown report services, and other gs-quant functions. This is not a claim of full upstream compatibility. |

Observation window size must be positive and its ramp at most the number of rows; for `moving_average`, Pearson `correlation`, `beta`, and `max_drawdown`, size larger than `x` returns an empty series (after rejecting `ramp > x.len()` with `Err(Error::Ramp)`). For `volatility`, size larger than the series returns empty when `annualization_factor` is `Some(...)`, and `Err(Error::AnnualizationFactor)` when `None`. For zero-mean `correlation`, size larger than the aligned non-NaN pair count (including `Window::Expanding` when `aligned.len() < x.len()`) returns `Err(Error::AnnualizationFactor)` unless an inner observation ramp `> aligned.len()` triggers `Err(Error::Ramp)` first. Calendar-day size must be positive. Empty `moving_average`, `volatility`, and left-input `correlation` and `beta` return empty before window or annualization validation; empty `max_drawdown` returns `Err(Error::Ramp)` when given a positive calendar-day ramp (`Window::Days { ramp > 0, .. }`, mapping upstream's `IndexError`) and empty otherwise; `returns` rejects lag zero even on empty input; `exponential_moving_average` validates `0.0 <= beta < 1.0` even on empty input, returning `Err(Error::Beta)` for negative values, values `>= 1.0`, `±Inf`, and `NaN`. A larger-than-series lag gives all NaNs. Singleton or unclassifiable spacing under automatic annualization returns `Err(Error::AnnualizationFactor)`. Use `Window::Observations { size: n, ramp: n }` for the upstream integer-window default ramp and `Window::Days { size: n, ramp: n }` for its day-string default. Public errors are `Error` variants.

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

Standalone offline `beta` example for an external Cargo project:

```rust
use quantus_math::{beta, returns, Date, Error, ReturnKind, Series, Window};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();
    let daily = vec![
        d(2019, 1, 1),
        d(2019, 1, 2),
        d(2019, 1, 3),
        d(2019, 1, 4),
        d(2019, 1, 7),
        d(2019, 1, 8),
    ];
    let stock = Series::try_new(
        daily.clone(),
        vec![100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792],
    )
    .unwrap();
    let market = Series::try_new(
        daily.clone(),
        vec![100.0, 102.0, 104.0, 101.0, 100.95, 100.0],
    )
    .unwrap();

    // 1. Default expanding window on two close-price series (deterministic on re-run)
    let def1 = beta(&stock, &market, Window::Expanding, true).unwrap();
    let def2 = beta(&stock, &market, Window::Expanding, true).unwrap();
    assert_eq!(def1.dates(), def2.dates());
    assert!(def1
        .values()
        .iter()
        .zip(def2.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(def1.values()[..3].iter().all(|v| v.is_nan()));
    assert!((def1.values()[5] - 0.572201).abs() < 1e-6);

    // 2. Precomputed simple returns: observation window vs calendar-day window
    let r_stock = returns(&stock, 1, ReturnKind::Simple).unwrap();
    let r_market = returns(&market, 1, ReturnKind::Simple).unwrap();
    let obs = beta(&r_stock, &r_market, Window::Observations { size: 2, ramp: 0 }, false).unwrap();
    let cal = beta(&r_stock, &r_market, Window::Days { size: 2, ramp: 2 }, false).unwrap();
    assert!((obs.values()[3] - 0.8255252918287954).abs() < 1e-12);
    assert_eq!(cal.dates(), &daily[2..]);
    assert!(cal.values()[0].is_nan() && cal.values()[2].is_nan());
    assert!((cal.values()[3] - (-2.24327163719368)).abs() < 1e-12);

    // 3. Empty -> empty, constant benchmark or disjoint dates -> NaN, invalid window -> Error
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(beta(&empty, &market, Window::Expanding, true)
        .unwrap()
        .values()
        .is_empty());
    let flat = Series::try_new(daily.clone(), vec![0.1; 6]).unwrap();
    assert!(beta(&r_stock, &flat, Window::Days { size: 3, ramp: 0 }, false)
        .unwrap()
        .values()
        .iter()
        .all(|v| v.is_nan()));
    let unmatched = Series::try_new(vec![d(2019, 2, 1), d(2019, 2, 2)], vec![0.01, 0.02]).unwrap();
    assert!(beta(&r_stock, &unmatched, Window::Observations { size: 2, ramp: 0 }, false)
        .unwrap()
        .values()
        .iter()
        .all(|v| v.is_nan()));
    assert_eq!(
        beta(&stock, &market, Window::Observations { size: 0, ramp: 0 }, true),
        Err(Error::Window)
    );
    assert_eq!(
        beta(&stock, &market, Window::Observations { size: 12, ramp: 12 }, true),
        Err(Error::Ramp)
    );

    println!(
        "expanding={} obs={} cal={}",
        def1.values()[5],
        obs.values()[3],
        cal.values()[3]
    );
}
```

Standalone offline `max_drawdown` example for an external Cargo project:

```rust
use quantus_math::{max_drawdown, Date, Error, Series, Window};

fn main() {
    let d = |day| Date::new(2019, 1, day).unwrap();
    // Close prices with a weekend trading gap between Jan 4 and Jan 7
    let dates = vec![d(1), d(2), d(3), d(4), d(7), d(8)];
    let prices = Series::try_new(dates.clone(), vec![1.0, 5.0, 5.0, 4.0, 4.0, 1.0]).unwrap();

    // 1. Default expanding window: re-running immediately gives identical dates and values
    let def1 = max_drawdown(&prices, Window::Expanding).unwrap();
    let def2 = max_drawdown(&prices, Window::Expanding).unwrap();
    assert_eq!(def1.dates(), def2.dates());
    assert!(def1
        .values()
        .iter()
        .zip(def2.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!((def1.values()[3] - (-0.2)).abs() < 1e-12);
    assert!((def1.values()[5] - (-0.8)).abs() < 1e-12);

    // 2. Trading gap comparison: 2 observations vs 2 calendar days diverge on Jan 7 (-0.2 vs 0.0)
    let obs = max_drawdown(&prices, Window::Observations { size: 2, ramp: 0 }).unwrap();
    let cal = max_drawdown(&prices, Window::Days { size: 2, ramp: 0 }).unwrap();
    let cal_ramp = max_drawdown(&prices, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert!((obs.values()[4] - (-0.2)).abs() < 1e-12);
    assert_eq!(cal.values()[4], 0.0);
    assert!((obs.values()[5] - (-0.75)).abs() < 1e-12);
    assert!((cal.values()[5] - (-0.75)).abs() < 1e-12);
    assert_eq!(cal_ramp.dates(), &dates[2..]);

    // 3. Empty input, missing prices (NaN), and invalid ramp / window errors
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(max_drawdown(&empty, Window::Expanding).unwrap().values().is_empty());
    assert!(max_drawdown(&empty, Window::Days { size: 2, ramp: 0 }).unwrap().values().is_empty());
    assert_eq!(
        max_drawdown(&empty, Window::Days { size: 2, ramp: 2 }),
        Err(Error::Ramp)
    );

    let missing = Series::try_new(
        vec![d(1), d(2), d(4), d(7), d(8)],
        vec![5.0, f64::NAN, 2.0, f64::NAN, 1.0],
    )
    .unwrap();
    let miss_cal = max_drawdown(&missing, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert!(miss_cal.values()[3].is_nan());
    assert_eq!(miss_cal.values()[4], 0.0);

    assert_eq!(
        max_drawdown(&prices, Window::Observations { size: 10, ramp: 10 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        max_drawdown(&prices, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );

    println!(
        "expanding={} obs_jan7={} cal_jan7={} last={}",
        def1.values()[5],
        obs.values()[4],
        cal.values()[4],
        cal.values()[5]
    );
}
```

Standalone offline `exponential_moving_average` example for an external Cargo project:

```rust
use quantus_math::{exponential_moving_average, Date, Error, Series};

fn main() {
    let d = |m, day| Date::new(2020, m, day).unwrap();
    // Irregular close dates crossing leap day 2020-02-29 (EMA decays by row order, not calendar days)
    let dates = vec![d(2, 27), d(2, 28), d(3, 2), d(3, 7), d(3, 14), d(3, 23)];
    let closes = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // 1. Default weight (beta = 0.75 when None): every input date gets its EMA
    let def = exponential_moving_average(&closes, None).unwrap();
    assert_eq!(def.dates(), dates.as_slice());
    let expected_def = [3.0, 2.75, 2.8125, 2.359375, 2.51953125, 3.3896484375];
    for (&actual, &expected) in def.values().iter().zip(&expected_def) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // 2. Custom weight (beta = 0.6) on the same irregular dates
    let b06 = exponential_moving_average(&closes, Some(0.6)).unwrap();
    assert_eq!(b06.dates(), dates.as_slice());
    let expected_06 = [3.0, 2.6, 2.76, 2.056, 2.4336, 3.86016];
    for (&actual, &expected) in b06.values().iter().zip(&expected_06) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // 3. Missing values (NaN) and ±Inf: leading NaN and carried-forward dates match across re-runs
    let gapped = Series::try_new(
        dates.clone(),
        vec![f64::NAN, f64::INFINITY, 3.0, f64::NEG_INFINITY, 6.0, f64::NAN],
    )
    .unwrap();
    let run1 = exponential_moving_average(&gapped, None).unwrap();
    let run2 = exponential_moving_average(&gapped, None).unwrap();
    assert_eq!(run1.dates(), run2.dates());
    assert!(run1
        .values()
        .iter()
        .zip(run2.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(run1.values()[0].is_nan() && run1.values()[1].is_nan());
    assert_eq!(run1.values()[2], 3.0);
    assert_eq!(run1.values()[3], 3.0);
    assert!((run1.values()[4] - (51.0 / 13.0)).abs() < 1e-12);
    assert_eq!(run1.values()[5].to_bits(), run1.values()[4].to_bits());

    // 4. Invalid weights (1.0, negative, NaN) return Err(Error::Beta), including on empty input
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(exponential_moving_average(&empty, None)
        .unwrap()
        .values()
        .is_empty());
    for bad in [1.0, -0.1, f64::NAN] {
        assert_eq!(exponential_moving_average(&closes, Some(bad)), Err(Error::Beta));
        assert_eq!(exponential_moving_average(&empty, Some(bad)), Err(Error::Beta));
    }

    println!(
        "default_last={} beta06_last={} gapped_last={}",
        def.values()[5],
        b06.values()[5],
        run1.values()[5]
    );
}
```

Parity: `python3 parity/check.py --replay`, `python3 parity/check-volatility.py --replay`, `python3 parity/check-correlation.py --replay`, `python3 parity/check-beta.py --replay`, `python3 parity/check-drawdown.py --replay`, and `python3 parity/check-ema.py --replay` build the offline replay binary and check frozen unrounded Python outputs. [The assertion map](parity/MAP.md) records the covered cases and exclusions. To regenerate vectors, use the pinned Python environment and an independently verified checkout of the upstream SHA:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-volatility.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-correlation.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-beta.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-drawdown.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-ema.py /path/to/gs-quant
```

Generation imports the upstream Python implementation; replay does not. Compare a regenerated file and its hash before accepting any change. Upstream LICENSE, NOTICE and NOTICE.txt are preserved; source attribution is in `src/lib.rs`.
