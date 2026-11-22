비공식 Rust 포팅 — 원본 goldmansachs/gs-quant, Goldman Sachs 와 무관

Unofficial Rust port of goldmansachs/gs-quant; not affiliated with, endorsed or sponsored by Goldman Sachs.

Quantus Analytics is a unified repository for financial calculations. Its first package, `quantus-math`, provides offline, dependency-free moving averages, exponential moving averages, smoothed moving averages, relative strength index, moving average convergence divergence (MACD), returns, price level reconstruction (`prices`), realized volatility, correlation, beta, and maximum drawdown as a slice of [gs-quant](https://github.com/goldmansachs/gs-quant), based on upstream commit `70220e300524d54bd273be0f4c485d53b3780482`. Prices (including reconstructed levels from `prices`), moving averages, exponential moving averages, smoothed moving averages, MACD, and absolute returns use the caller's price units; simple returns, logarithmic returns, correlation, beta, and maximum drawdown are dimensionless ratios (not percent; e.g. `-0.2` for a 20% drawdown); relative strength index (RSI) is returned on a `0`–`100` scale; volatility is returned in percent (e.g. `20.0` for 20%). No network, market data, or internal engine is involved.

| Feature | Supported subset |
|---|---|
| Index | Strictly increasing unique Gregorian dates, years 1–9999; no times or time zones. Length mismatch and duplicate or unsorted dates are errors. |
| Moving average | Expanding default (`Window::Expanding`), observation window with observation ramp, or calendar-day window with calendar-day ramp. Days use `(date - size, date]`; ramp keeps dates at or after first date + ramp. NaN values are ignored; a window with no non-NaN values gives NaN. |
| Exponential moving average | Reuses `Series`, `Date`, and `Error::Beta`. `exponential_moving_average(x, beta: Option<f64>)` uses upstream default `beta = 0.75` when `None`, and decays over observation (row) order rather than calendar-day spacing, matching upstream `x.ewm(alpha=1 - beta, adjust=False).mean()`. Leading `NaN` and `±Inf` values emit `NaN` until the first finite observation; subsequent `NaN` and `±Inf` values are treated as missing, carrying forward the prior average while decaying its weight across missing rows when the next finite observation arrives. Equal finite values preserve the prior value and sign (including `-0.0 == +0.0`, even at `beta = 0.0`). Valid weights satisfy `0.0 <= beta < 1.0` (`-0.0` is accepted as `0.0`); negative values, values `>= 1.0`, `±Inf`, and `NaN` return `Err(Error::Beta)` even on empty input. |
| Smoothed moving average | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `Date`, ramp, and `Error`. `smoothed_moving_average(x, window)` mirrors upstream `gs_quant/timeseries/technicals.py:smoothed_moving_average` on a `datetime.date` index: initializes the first post-ramp value from `apply_ramp(mean(x, Window(size, 0)), w).iloc[0]` (skipping `NaN` and `±Inf` in that initial mean window, matching pandas `rolling().mean()`), slices `x` by the ramp before iterating, and applies `out[i] = ((n - 1) * out[i - 1] + x[i]) / n`. For `Window::Expanding` (`n = x.dates().len()`, ramp `0`) and `Window::Observations`, `n` is the normalized window size; for `Window::Days`, `n` is the count of post-ramp observations in `(date - size, date]`. Note that upstream calendar-day windows compare `(x.index[i] - window_size).date()`, which raises `TypeError` on a pandas `DatetimeIndex` and works only on a `datetime.date` index; Rust's date-only `Series` matches the `datetime.date` index behavior. As with `max_drawdown`, calendar-day parity with Python holds only within the pinned pandas `DateOffset` date range (whereas Rust evaluates integer day-count differences between `Date` ordinals across years 1–9999 without constructing pandas timestamps or `DateOffset` values). In the recurrence loop, `NaN` and `±Inf` propagate via IEEE-754 arithmetic (opposite-sign `±Inf` or `0.0 * ±Inf` yields `NaN`), and signed zeros follow IEEE-754 arithmetic (`[-0.0, 0.0, -0.0, -0.0]` under `Window::Observations { size: 2, ramp: 0 }` yields `[-0.0, +0.0, +0.0, +0.0]`). Empty input returns empty for `Expanding`, `Observations`, and `Days { ramp: 0 }`, and returns `Err(Error::Ramp)` for `Days { ramp > 0 }` (mapping upstream's `IndexError`). On non-empty input, `size == 0` returns `Err(Error::Window)` (including `Window::Days { size: 0, .. }`, whereas upstream `dt.date` divides by zero), observation `ramp > len` returns `Err(Error::Ramp)`, and oversize `size > len` (with valid ramp) or a ramp leaving no rows returns empty. |
| Relative strength index | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `smoothed_moving_average`, `Date`, ramp, and `Error`. `relative_strength_index(x, window: Option<Window>)` defaults to `Window::Observations { size: 14, ramp: 14 }` when `None` and follows the exact order of upstream `gs_quant/timeseries/technicals.py:relative_strength_index` on a `datetime.date` index: normalizes `window` against original `x` first (`Window::Expanding` becomes `size = x.dates().len(), ramp = 0`, which always yields an empty `Series` because `gains`/`losses` have length `x.dates().len().saturating_sub(1)`), splits one-period differences `diff(x, 1)[1:]` into non-negative `gains` and `losses`, calls `smoothed_moving_average` on each, and evaluates `100.0` when `avg_losses == 0.0` or `100.0 - 100.0 / (1.0 + avg_gains / avg_losses)` otherwise. Under this upstream rule, constant prices (`gains == 0`, `losses == 0`) take the `avg_losses == 0.0` branch and return **`100.0`** (monotonically increasing prices also return `100.0`; monotonically decreasing prices return `0.0`). Because upstream delegates calendar-day windows to `smoothed_moving_average`, which compares `(x.index[i] - window_size).date()`, upstream raises `TypeError` on a pandas `DatetimeIndex` and works only on a `datetime.date` index; Rust's date-only `Series` matches the `datetime.date` index behavior. As with `max_drawdown` and `smoothed_moving_average`, calendar-day parity with Python holds only within the pinned pandas `DateOffset` date range (whereas Rust evaluates integer day-count differences between `Date` ordinals across years 1–9999 without constructing pandas timestamps or `DateOffset` values). Mid-series `NaN` and `±Inf` propagate through `diff` and `smoothed_moving_average` (e.g. a `+Inf` jump with `avg_losses == 0.0` emits `100.0` on that row and `NaN` once `Inf / Inf` occurs). Empty input returns empty for `None`, `Expanding`, `Observations`, and `Days { ramp: 0 }`, and returns `Err(Error::Ramp)` for `Days { ramp > 0 }` (mapping upstream's `IndexError`). On non-empty input, `size == 0` returns `Err(Error::Window)` (including `Window::Days { size: 0, .. }`, whereas upstream `dt.date` divides by zero), observation `ramp > x.dates().len()` or `ramp > len(gains)` (when `gains` is non-empty) returns `Err(Error::Ramp)`, length-1 input returns empty for valid windows with `ramp <= 1` (or `Days { ramp: 0 }`) and `Err(Error::Ramp)` for `Days { ramp > 0 }` or observation `ramp > 1`, and length-2 input with `Window::Observations { size: 1, ramp: 1 }` returns empty. |
| MACD | Reuses `Series`, `Date`, and `Error::Span`. `macd(x, m: Option<usize>, n: Option<usize>, s: Option<usize>)` defaults to `m = 12`, `n = 26`, `s = 1` when `None`, and mirrors upstream `gs_quant/timeseries/technicals.py:macd` (lines 299–344) and `algebra.py:subtract` (`Interpolate.STEP` = `ffill().bfill()`). All three EWM stages decay over observation (row) order rather than calendar-day spacing (`adjust = false`, `com = (span - 1) / 2`, `alpha = 1 / (1 + com)`), preserving irregular dates including leap day `2020-02-29` verbatim. Span validation runs before empty-input handling: any `m == 0`, `n == 0`, or `s == 0` returns `Err(Error::Span)` even on an empty `Series`; any positive combination (including `m == n`, `m > n`, `m = 1`, `n = 1`, `s = 1`) is accepted. Because upstream calls `subtract(a, b)` with default `Interpolate.STEP` (`ffill().bfill()`), leading `NaN` or `±Inf` rows followed by a finite observation are backfilled with that first finite value in both `a` and `b`, producing `+0.0` in `diff = a_filled - b_filled` at every leading missing index before the third EWM (`span = s`) runs (so for `s > 1`, the first non-zero diff is smoothed against `+0.0`); if `x` has no finite values at all, the result is all `NaN`. If `a - b` overflows to `±Inf`, the third EWM treats that non-finite entry as missing, carrying forward the prior finite average while decaying `old_wt`. Finite equal values (including constant `-0.0` or mixed signed zeros) and finite length-1 series emit `+0.0`; empty input with valid spans returns empty. |
| Returns | Positive integer observation lag, SIMPLE, LOGARITHMIC and ABSOLUTE. NaN propagates. Logarithmic returns reject nonpositive prices. Division by zero follows floating-point arithmetic. |
| Prices | Reuses `Series`, `ReturnKind`, `Date`, and `Error::Overflow`. `prices(x, initial: Option<f64>, kind: Option<ReturnKind>)` reconstructs price levels from a returns series with defaults `initial = 1.0` and `kind = ReturnKind::Simple`, mirroring upstream `gs_quant/timeseries/econometrics.py:prices` (lines 353–421) and `statistics.py:{product,sum_}`. Round-trips `returns(px, 1, kind)` back to `px` via `prices(&r, Some(px[0]), Some(kind))` across `Simple`, `Logarithmic`, and `Absolute` while preserving irregular dates including leap day `2020-02-29`. Empty input returns empty regardless of `initial` (even `NaN`/`±Inf`) or `kind`. All modes skip `NaN` and `±Inf` (matching pandas 2.3.3 `rolling(len, 0)`): `ReturnKind::Simple` multiplies finite `1.0 + v` factors starting from `prod = 1.0` and emits `prod * initial` (so leading `NaN` emits `initial`, a `-1.0` return collapses subsequent finite products to `0.0`, and a later `< -1.0` return flips the sign to `-0.0`); `ReturnKind::Logarithmic` first returns `Err(Error::Overflow)` if any finite `v` has `v.exp().is_infinite()` (e.g. `800.0`, matching Python `math.exp` `OverflowError`), skips `+Inf` without error, maps `-Inf` and underflow (e.g. `-800.0`) to `0.0`, allows cumulative product overflow of finite exponentials (e.g. `[709.0, 709.0]` → `+Inf`), and emits `prod * initial`; `ReturnKind::Absolute` sums finite values (emitting `+0.0 + initial` before the first finite value, and accumulating from `-0.0` once a finite value appears so `-0.0` is preserved when all finite inputs and `initial` are `-0.0`). For `ReturnKind::Absolute`, parity applies to finite-sum inputs without overflow cancellation: when large finite values overflow the running sum to `±Inf` (e.g. `[1e308, 1e308, -1e308]` or `[1e308, 1e308, 1.0]`), upstream pandas `rolling().sum()` uses compensated Kahan summation where `(+Inf - +Inf)` yields `NaN`, whereas Rust uses plain `f64` summation (staying `+Inf`). |
| Volatility | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<ReturnKind>` (`Simple`, `Logarithmic`, `Absolute`, or `None` for precomputed returns). Standard sample volatility (`assume_zero_mean = false`, `n - 1` denominator, `NaN` for `< 2` non-NaN values) and zero-mean RMS (`assume_zero_mean = true`, `n` denominator, `NaN` for `0` non-NaN values). Explicit `Option<u32>` annualization factor (including `0`, which scales finite values to `0.0` and keeps `NaN`), or `None` for upstream automatic classification (`< 2.1d → 252`, `[6, 8) → 52`, `[14, 17) → 26`, `[25, 35) → 12`, `[85, 97) → 4`, `[360, 386) → 1`). Output is annualized in percent (`* sqrt(factor) * 100`). |
| Correlation | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<(ReturnKind, ReturnKind)>` (`Some((kx, ky))` computes lag-1 returns on `x` and `y` independently before date intersection; `None` treats both inputs as precomputed returns). Aligns common dates and drops any pair with a `NaN` before rolling (`Observations` counts aligned non-NaN pairs; `Days` uses `(date - size, date]`). Reindexes results onto `x`'s dates (unmatched left dates become `NaN`) and applies the outer ramp on `x`. Returns a dimensionless ratio (not percent). Sample Pearson (`assume_zero_mean = false`) yields `NaN` when a window has `< 2` pairs or zero variance. Zero-mean mode (`assume_zero_mean = true`) computes normalized inner products via upstream's `mean` + zero-mean `volatility` pipeline on the aligned pairs: both the inner window/ramp (on aligned pairs) and outer ramp (on `x`) apply, so dates masked by the inner ramp remain `NaN` even if they survive the outer ramp; empty, singleton, irregular-spacing, or oversize-window aligned pairs (including `Window::Expanding` when aligned count `< x.len()`) return `Err(Error::AnnualizationFactor)`, and inner observation ramp `> aligned.len()` returns `Err(Error::Ramp)`. |
| Beta | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `ReturnKind::Simple` (internally when `prices = true`), `Error`, and outer ramp. When `prices = true`, computes lag-1 simple returns independently on `x` and `benchmark`; when `prices = false`, treats both series as precomputed returns. For `Window::Expanding` (size = `x.dates().len()`) and `Window::Observations`, computes rolling sample covariance (`ddof = 1`, skipping `NaN` pairs) over the outer-joined date union divided by `benchmark`'s own rolling sample variance (`ddof = 1`, skipping `NaN`s) aligned by date—without pooling the two sample sets. For `Window::Days`, computes sample covariance divided by benchmark variance over `(date - size, date]` on the date intersection, propagating any `NaN` inside the window. Masks the first 3 positions of the intermediate result (`union` for observations/expanding, `intersection` for days) to `NaN`, reindexes onto `x`'s dates (`NaN` for unmatched dates), and applies the outer ramp. An exactly constant finite benchmark window yields `NaN` (zero denominator). |
| Max drawdown | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `Date`, ramp, and `Error`. Two-stage rolling peak-to-trough ratio: computes the rolling maximum price, forms `price / rolling_max - 1`, and takes the rolling minimum of those ratios over the same window before applying the ramp (e.g. `-0.2` for a 20% drawdown). `Window::Expanding` uses `x.dates().len()` with ramp `0`; `Window::Observations` skips `NaN` and `±Inf` in both rolling stages (matching upstream pandas `rolling().max()` / `rolling().min()`); `Window::Days` uses `(date - size, date]` in both stages without interpolating holidays, skipping `NaN` while retaining `±Inf` (matching upstream pandas `Series.max()` / `Series.min()`). All-NaN windows and `0 / 0` stay `NaN`. Empty input returns empty for `Expanding`, `Observations`, and `Days { ramp: 0 }`, and returns `Err(Error::Ramp)` for `Days { ramp > 0 }` (mapping upstream's `IndexError` when applying a positive calendar ramp to an empty date index). For `Window::Days`, date parity with Python holds only when the input dates, every left window boundary (`date - size` days), and the ramp cutoff (`first_date + ramp` days) are representable and computable with the pinned pandas `DateOffset`. Rust evaluates calendar windows and ramps via integer day-count differences between `Date` ordinals (without constructing pandas timestamps or `DateOffset` values), so extremes that raise Python date-range or `DateOffset` errors (`OverflowError` / `OutOfBoundsDatetime`) can still produce normal values or an empty `Series` in Rust—for example, on `2024-01-01`/`2024-01-05` with prices `[2.0, 1.0]`, an ordinary window such as `size: 100, ramp: 0` matches Python (`[0.0, -0.5]`), whereas `size: 1_000_000` returns `[0.0, -0.5]` and `ramp: 1_000_000` returns empty in Rust while Python errors; similarly, near pandas's lower timestamp limit, `1677-09-22`/`1677-09-23` with `size: 2` succeeds in Rust (`[0.0, -0.5]`) even though the left boundary `date - 2` days is out of range for pandas. |
| Numerical precision & non-finite inputs | Parity tolerance `1e-12 + 1e-10 * |expected|` applies to well-conditioned finite inputs. For `volatility`, `correlation`, and `beta`, inputs with `±Inf` are excluded from compatibility vectors: Rust follows IEEE-754 arithmetic (`NaN` for sample volatility, Pearson correlation, and beta, `+Inf` for RMS, `NaN` for zero-mean `Inf / Inf`), whereas upstream pandas `rolling()` and numpy `rolling_std()` diverge on `±Inf`. For `max_drawdown`, `±Inf` (including `-Inf` produced by a negative price following a `+0.0` peak) matches upstream: integer/expanding windows skip `±Inf` in both rolling stages while calendar-day windows retain `±Inf`, except when a calendar-day window's rolling peak ties across mixed-sign zeros (`+0.0` and `-0.0`). On mixed-sign zero peak ties in `Window::Days`, Rust keeps the last zero in the window (updating on `!(cur > v)`), so a negative price divides by that zero's sign (`-Inf` when the last zero is `+0.0`, `+Inf` when it is `-0.0`); even within the same pinned Python/NumPy/pandas environment, the resulting `±Inf` sign for mixed-sign zeros can vary depending on NumPy CPU optimizations. For constant non-power-of-two float series (e.g. `0.01`, `0.1`), Rust detects finite constant windows before centering and returns `NaN` (zero variance / zero denominator), whereas upstream pandas 2.3.3 observation-window `rolling().corr()` can emit `±Inf` from covariance rounding residue divided by zero standard deviation (e.g. on `0.01`, or `0.1` vs varying), upstream calendar-day `Series.corr` (`np.corrcoef`) can emit rounding-residue `1.0` (constant/constant) or `0.0` (constant/varying) when `sum(x) / n != x` (e.g. three `0.1` values), and upstream calendar-day `beta` (`np.cov` / `np.var`) emits `±21.333333333333332` (`±64/3`) on constant `0.1` over a 6-day window (`constant-calendar` vector) due to floating-point rounding residue in the denominator variance. For inputs with very large constant offsets relative to spread (e.g. `1e9 + 0.1`, where `f64` input ULP is `~1.2e-7`), Rust uses two-pass centered sums (`sum((x - mean)^2)` / `sum((x - mean_x)(y - mean_y))`), avoiding catastrophic cancellation while upstream pandas observation-window and numpy day-window paths diverge by `~6e-7` relative error (outside the `1e-10` parity bound). For `exponential_moving_average`, `±Inf` inputs match upstream pandas (treated as missing like `NaN`), and signed zeros match upstream bit-for-bit; for `beta = NaN`, upstream `gs-quant` documents `0 <= beta < 1` (`technicals.py:265-266`) while pandas `get_center_of_mass` checks `alpha <= 0 or alpha > 1` and lets `NaN` slip through (producing alternating artifacts such as `[2.0, NaN, 4.0]` on `[2.0, 3.0, 4.0]`), whereas Rust strictly enforces `[0.0, 1.0)` and returns `Err(Error::Beta)`. For `prices`, `NaN` and `±Inf` skipping, `-1.0` collapse, signed zeros, `ReturnKind::Logarithmic` `Error::Overflow` on finite `exp` overflow, log underflow, and cumulative product overflow all match upstream; only extreme `ReturnKind::Absolute` inputs whose finite running sum overflows `f64` diverge from pandas's Kahan summation (`+Inf` in Rust vs `NaN` in pandas). |
| Not supported | Date-based return lag, months/weeks (`1m`, `3m`, `1w`), mixed-unit ramps, duplicate or unsorted indices, `DatetimeIndex` / time zones, non-date indices (`TypeError` in upstream `max_drawdown`), pandas nullable values distinct from NaN, non-simple return kinds for `beta`, volatility/variance/correlation swap wrappers (`vol_swap_volatility`, `corr_swap_correlation`), service-level portfolio beta (`measures_portfolios.py`) and drawdown report services, and other gs-quant functions. This is not a claim of full upstream compatibility. |

Observation window size must be positive and its ramp at most the number of rows; for `moving_average`, `smoothed_moving_average`, `relative_strength_index` (against `gains` of length `x.len().saturating_sub(1)` after initial normalization on `x`), Pearson `correlation`, `beta`, and `max_drawdown`, size larger than the effective series returns an empty series (after rejecting an invalid observation ramp with `Err(Error::Ramp)`). For `volatility`, size larger than the series returns empty when `annualization_factor` is `Some(...)`, and `Err(Error::AnnualizationFactor)` when `None`. For zero-mean `correlation`, size larger than the aligned non-NaN pair count (including `Window::Expanding` when `aligned.len() < x.len()`) returns `Err(Error::AnnualizationFactor)` unless an inner observation ramp `> aligned.len()` triggers `Err(Error::Ramp)` first. Calendar-day size must be positive. Empty `moving_average`, `volatility`, and left-input `correlation` and `beta` return empty before window or annualization validation; empty `max_drawdown`, `smoothed_moving_average`, and `relative_strength_index` return `Err(Error::Ramp)` when given a positive calendar-day ramp (`Window::Days { ramp > 0, .. }`, mapping upstream's `IndexError`) and empty otherwise; `returns` rejects lag zero even on empty input; `exponential_moving_average` validates `0.0 <= beta < 1.0` even on empty input, returning `Err(Error::Beta)` for negative values, values `>= 1.0`, `±Inf`, and `NaN`; `macd` validates `m >= 1`, `n >= 1`, `s >= 1` even on empty input, returning `Err(Error::Span)` if any span is `0`; `prices` returns empty on empty input regardless of `initial` or `kind`, and returns `Err(Error::Overflow)` in `ReturnKind::Logarithmic` when any finite `v` overflows `v.exp()`. A larger-than-series lag gives all NaNs. Singleton or unclassifiable spacing under automatic annualization returns `Err(Error::AnnualizationFactor)`. Use `Window::Observations { size: n, ramp: n }` for the upstream integer-window default ramp and `Window::Days { size: n, ramp: n }` for its day-string default. Public errors are `Error` variants.

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

Standalone offline `smoothed_moving_average` example for an external Cargo project:

```rust
use quantus_math::{smoothed_moving_average, Date, Error, Series, Window};

fn main() {
    let d_reg = |day| Date::new(2019, 1, day).unwrap();
    let reg_dates: Vec<_> = (1..=6).map(d_reg).collect();
    let closes = Series::try_new(reg_dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // 1. Default expanding window and observation window on daily close prices
    let exp = smoothed_moving_average(&closes, Window::Expanding).unwrap();
    let obs = smoothed_moving_average(&closes, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(exp.dates(), reg_dates.as_slice());
    assert!((exp.values()[5] - (24791.0 / 7776.0)).abs() < 1e-12);
    assert_eq!(obs.values(), &[3.0, 2.5, 2.75, 1.875, 2.4375, 4.21875]);

    // 2. Calendar-day windows (size 2 with ramp 2 vs ramp 0) on regular and leap-day irregular dates
    let reg_2d = smoothed_moving_average(&closes, Window::Days { size: 2, ramp: 2 }).unwrap();
    let reg_2d_r0 = smoothed_moving_average(&closes, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert_eq!(reg_2d.dates(), &reg_dates[2..]);
    assert_eq!(reg_2d.values(), &[2.5, 1.75, 2.375, 4.1875]);
    assert_eq!(reg_2d_r0.values(), obs.values());

    let d_leap = |m, day| Date::new(2020, m, day).unwrap();
    let irreg_dates = vec![
        d_leap(2, 27),
        d_leap(2, 28),
        d_leap(3, 2),
        d_leap(3, 7),
        d_leap(3, 14),
        d_leap(3, 23),
    ];
    let irreg = Series::try_new(irreg_dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let irreg_2d = smoothed_moving_average(&irreg, Window::Days { size: 2, ramp: 2 }).unwrap();
    let irreg_2d_r0 = smoothed_moving_average(&irreg, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert_eq!(irreg_2d.dates(), &irreg_dates[2..]);
    assert_eq!(irreg_2d.values(), &[3.0, 1.0, 3.0, 6.0]);
    assert_eq!(irreg_2d_r0.dates(), irreg_dates.as_slice());
    assert_eq!(irreg_2d_r0.values(), &[3.0, 2.5, 3.0, 1.0, 3.0, 6.0]);

    // 3. NaN, ±Inf, signed zero (-0.0), and oversize windows: two back-to-back runs match bit-for-bit
    let edge_cases = [
        (
            Series::try_new(reg_dates.clone(), vec![2.0, 4.0, f64::NAN, 1.0, 3.0, 6.0]).unwrap(),
            Window::Observations { size: 2, ramp: 0 },
        ),
        (
            Series::try_new(reg_dates.clone(), vec![2.0, 4.0, f64::INFINITY, 1.0, 3.0, 6.0])
                .unwrap(),
            Window::Observations { size: 2, ramp: 0 },
        ),
        (
            Series::try_new(reg_dates[..4].to_vec(), vec![-0.0, 0.0, -0.0, -0.0]).unwrap(),
            Window::Observations { size: 2, ramp: 0 },
        ),
        (
            closes.clone(),
            Window::Observations { size: 10, ramp: 0 },
        ),
    ];
    for (series, window) in &edge_cases {
        let a = smoothed_moving_average(series, *window).unwrap();
        let b = smoothed_moving_average(series, *window).unwrap();
        assert_eq!(a.dates(), b.dates());
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    let nz = smoothed_moving_average(&edge_cases[2].0, edge_cases[2].1).unwrap();
    assert!(nz.values()[0] == 0.0 && nz.values()[0].is_sign_negative());
    assert!(nz.values()[1..].iter().all(|v| *v == 0.0 && v.is_sign_positive()));
    assert!(smoothed_moving_average(&edge_cases[3].0, edge_cases[3].1)
        .unwrap()
        .values()
        .is_empty());

    // 4. Empty input and invalid windows (size == 0, ramp > len, empty + positive calendar ramp)
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(smoothed_moving_average(&empty, Window::Expanding)
        .unwrap()
        .values()
        .is_empty());
    assert!(
        smoothed_moving_average(&empty, Window::Days { size: 2, ramp: 0 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert_eq!(
        smoothed_moving_average(&empty, Window::Days { size: 2, ramp: 2 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        smoothed_moving_average(&closes, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        smoothed_moving_average(&closes, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        smoothed_moving_average(&closes, Window::Observations { size: 10, ramp: 10 }),
        Err(Error::Ramp)
    );

    println!(
        "expanding_last={} obs2_last={} reg2d_last={} irreg2d_last={}",
        exp.values()[5],
        obs.values()[5],
        reg_2d.values()[3],
        irreg_2d.values()[3]
    );
}
```

Standalone offline `relative_strength_index` example for an external Cargo project:

```rust
use quantus_math::{relative_strength_index, Date, Error, Series, Window};

fn main() {
    let d = |m, day| Date::new(2020, m, day).unwrap();

    // 1. Default window (None = Observations { size: 14, ramp: 14 }) and specified observation window
    let spx_dates = vec![
        d(1, 2), d(1, 3), d(1, 6), d(1, 7), d(1, 8), d(1, 9), d(1, 10), d(1, 13),
        d(1, 14), d(1, 15), d(1, 16), d(1, 17), d(1, 21), d(1, 22), d(1, 23), d(1, 24),
        d(1, 27), d(1, 28), d(1, 29), d(1, 30), d(1, 31), d(2, 3),
    ];
    let spx = Series::try_new(
        spx_dates.clone(),
        vec![
            3257.8501, 3234.8501, 3246.28, 3237.1799, 3253.05, 3274.7, 3265.3501, 3288.1299,
            3283.1499, 3289.29, 3316.8101, 3329.6201, 3320.79, 3321.75, 3325.54, 3295.47,
            3243.6299, 3276.24, 3273.3999, 3283.6599, 3225.52, 3248.9199,
        ],
    )
    .unwrap();
    let rsi_default = relative_strength_index(&spx, None).unwrap();
    assert_eq!(rsi_default.dates(), &spx_dates[15..]);
    assert!((rsi_default.values()[0] - 66.35898836410378).abs() < 1e-12);
    assert!((rsi_default.values()[6] - 50.617629713259795).abs() < 1e-12);

    let dates10: Vec<_> = (1..=10).map(|day| d(1, day)).collect();
    let mixed = Series::try_new(
        dates10.clone(),
        vec![10.0, 12.0, 11.0, 13.0, 13.0, 9.0, 10.5, 14.0, 12.0, 15.0],
    )
    .unwrap();
    let rsi_obs3 =
        relative_strength_index(&mixed, Some(Window::Observations { size: 3, ramp: 3 })).unwrap();
    assert_eq!(rsi_obs3.dates(), &dates10[4..]);
    assert!((rsi_obs3.values()[0] - (200.0 / 3.0)).abs() < 1e-12);
    assert!((rsi_obs3.values()[5] - 70.00547345374932).abs() < 1e-12);

    // 2. Calendar-day windows: 3d on daily closes and 4d on irregular dates crossing leap day 2020-02-29
    let rsi_3d = relative_strength_index(&mixed, Some(Window::Days { size: 3, ramp: 3 })).unwrap();
    assert_eq!(rsi_3d.dates(), &dates10[4..]);
    assert!((rsi_3d.values()[5] - 66.5086887835703).abs() < 1e-12);

    let irreg_dates = vec![
        d(2, 27), d(2, 28), d(3, 2), d(3, 4), d(3, 7), d(3, 10), d(3, 14), d(3, 20),
    ];
    let irreg = Series::try_new(
        irreg_dates.clone(),
        vec![10.0, 12.0, 11.0, 14.0, 12.0, 12.0, 15.0, 9.0],
    )
    .unwrap();
    let irreg_4d =
        relative_strength_index(&irreg, Some(Window::Days { size: 4, ramp: 4 })).unwrap();
    assert_eq!(irreg_4d.dates(), &irreg_dates[3..]);
    assert_eq!(irreg_4d.values(), &[75.0, 37.5, 37.5, 100.0, 0.0]);

    // 3. Monotonic inc (100), dec (0), constant (100 per upstream losses == 0 rule), mid NaN & ±Inf: deterministic
    let dates6 = dates10[..6].to_vec();
    let edge_cases = [
        (
            Series::try_new(dates10.clone(), (1..=10).map(f64::from).collect()).unwrap(),
            Window::Observations { size: 3, ramp: 3 },
        ),
        (
            Series::try_new(dates10.clone(), (1..=10).rev().map(f64::from).collect()).unwrap(),
            Window::Observations { size: 3, ramp: 3 },
        ),
        (
            Series::try_new(dates10.clone(), vec![5.0; 10]).unwrap(),
            Window::Observations { size: 3, ramp: 3 },
        ),
        (
            Series::try_new(dates6.clone(), vec![1.0, 2.0, f64::NAN, 4.0, 3.0, 5.0]).unwrap(),
            Window::Observations { size: 2, ramp: 0 },
        ),
        (
            Series::try_new(dates6.clone(), vec![1.0, 2.0, f64::INFINITY, 4.0, 3.0, 5.0]).unwrap(),
            Window::Observations { size: 2, ramp: 0 },
        ),
        (
            Series::try_new(dates6, vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 3.0, 5.0]).unwrap(),
            Window::Observations { size: 2, ramp: 0 },
        ),
    ];
    for (series, window) in &edge_cases {
        let run1 = relative_strength_index(series, Some(*window)).unwrap();
        let run2 = relative_strength_index(series, Some(*window)).unwrap();
        assert_eq!(run1.dates(), run2.dates());
        assert!(run1
            .values()
            .iter()
            .zip(run2.values())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert_eq!(
        relative_strength_index(&edge_cases[0].0, Some(edge_cases[0].1)).unwrap().values(),
        &[100.0; 6]
    );
    assert_eq!(
        relative_strength_index(&edge_cases[1].0, Some(edge_cases[1].1)).unwrap().values(),
        &[0.0; 6]
    );
    assert_eq!(
        relative_strength_index(&edge_cases[2].0, Some(edge_cases[2].1)).unwrap().values(),
        &[100.0; 6]
    );
    let r_inf = relative_strength_index(&edge_cases[4].0, Some(edge_cases[4].1)).unwrap();
    assert_eq!(&r_inf.values()[..2], &[100.0, 100.0]);
    assert!(r_inf.values()[2..].iter().all(|v| v.is_nan()));

    // 4. Empty, length-1, Window::Expanding, ramp > len(gains), size == 0, and empty + positive calendar ramp
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(relative_strength_index(&empty, None).unwrap().values().is_empty());
    assert_eq!(
        relative_strength_index(&empty, Some(Window::Days { size: 3, ramp: 3 })),
        Err(Error::Ramp)
    );
    let single = Series::try_new(vec![d(1, 1)], vec![5.0]).unwrap();
    assert!(
        relative_strength_index(&single, Some(Window::Observations { size: 1, ramp: 0 }))
            .unwrap()
            .values()
            .is_empty()
    );
    assert!(relative_strength_index(&mixed, Some(Window::Expanding))
        .unwrap()
        .values()
        .is_empty());
    assert_eq!(
        relative_strength_index(&mixed, Some(Window::Observations { size: 3, ramp: 10 })),
        Err(Error::Ramp)
    );
    assert_eq!(
        relative_strength_index(&mixed, Some(Window::Observations { size: 0, ramp: 0 })),
        Err(Error::Window)
    );
    assert_eq!(
        relative_strength_index(&mixed, Some(Window::Days { size: 0, ramp: 0 })),
        Err(Error::Window)
    );

    println!(
        "default_last={} obs3_last={} cal3d_last={} irreg4d_last={}",
        rsi_default.values()[6],
        rsi_obs3.values()[5],
        rsi_3d.values()[5],
        irreg_4d.values()[4]
    );
}
```

Standalone offline `macd` example for an external Cargo project:

```rust
use quantus_math::{macd, Date, Error, Series};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Default periods (None, None, None = m=12, n=26, s=1), specified (2, 3, 1), and signal smoothing (s=4, s=9)
    let dates10: Vec<_> = (1..=10).map(|day| d(2020, 1, day)).collect();
    let closes10 = Series::try_new(
        dates10.clone(),
        vec![100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 111.0, 110.0, 115.0],
    )
    .unwrap();
    let def = macd(&closes10, None, None, None).unwrap();
    let sig9 = macd(&closes10, Some(12), Some(26), Some(9)).unwrap();
    assert_eq!(def.dates(), dates10.as_slice());
    assert!((def.values()[9] - 2.881299976008549).abs() < 1e-12);
    assert!((sig9.values()[9] - 1.533517851885833).abs() < 1e-12);

    let dates6: Vec<_> = (1..=6).map(|day| d(2019, 1, day)).collect();
    let sample = Series::try_new(dates6.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let m2_3 = macd(&sample, Some(2), Some(3), Some(1)).unwrap();
    let m2_3_4 = macd(&sample, Some(2), Some(3), Some(4)).unwrap();
    assert!((m2_3.values()[5] - 0.6248713991769552).abs() < 1e-12);
    assert!((m2_3_4.values()[5] - 0.22544930041152286).abs() < 1e-12);

    // 2. Irregular dates including leap day 2020-02-29 (row-order EWM), m == n, m > n, and m=1/s=1
    let leap_dates = vec![d(2020, 2, 27), d(2020, 2, 28), d(2020, 2, 29), d(2020, 3, 2), d(2020, 3, 10)];
    let irreg = Series::try_new(leap_dates.clone(), vec![10.0, 12.0, 11.0, 15.0, 13.0]).unwrap();
    let irreg_macd = macd(&irreg, Some(2), Some(4), Some(3)).unwrap();
    assert_eq!(irreg_macd.dates(), leap_dates.as_slice());
    assert!((irreg_macd.values()[4] - 0.6150320987654324).abs() < 1e-12);
    assert!(macd(&sample, Some(10), Some(10), Some(1))
        .unwrap()
        .values()
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));
    assert!((macd(&sample, Some(5), Some(2), Some(3)).unwrap().values()[5] - (-0.5369084362139918)).abs() < 1e-12);

    // 3. Leading missing values (subtract STEP ffill/bfill -> +0.0 entering s=3), -0.0, overflow: deterministic
    let lead_nan = Series::try_new(
        dates10[..6].to_vec(),
        vec![f64::NAN, f64::NAN, 10.0, 20.0, 15.0, 25.0],
    )
    .unwrap();
    let neg_zero = Series::try_new(dates10[..5].to_vec(), vec![-0.0; 5]).unwrap();
    let overflow = Series::try_new(
        dates10[..5].to_vec(),
        vec![1.7e308, -1.7e308, 1.7e308, -1.7e308, 1.7e308],
    )
    .unwrap();
    for (series, m, n, s) in [
        (&lead_nan, Some(2), Some(4), Some(3)),
        (&neg_zero, Some(2), Some(5), Some(3)),
        (&overflow, Some(1), Some(20), Some(3)),
    ] {
        let run1 = macd(series, m, n, s).unwrap();
        let run2 = macd(series, m, n, s).unwrap();
        assert_eq!(run1.dates(), run2.dates());
        assert!(run1
            .values()
            .iter()
            .zip(run2.values())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let ln_s3 = macd(&lead_nan, Some(2), Some(4), Some(3)).unwrap();
    assert!(ln_s3.values()[..3].iter().all(|v| *v == 0.0 && v.is_sign_positive()));
    assert!((ln_s3.values()[3] - 1.3333333333333321).abs() < 1e-12);

    // 4. Zero spans return Error::Span on empty and non-empty inputs; valid empty and length-1 succeed
    let empty = Series::try_new(vec![], vec![]).unwrap();
    let single = Series::try_new(vec![d(2020, 1, 1)], vec![42.0]).unwrap();
    for series in [&empty, &sample] {
        assert_eq!(macd(series, Some(0), Some(26), Some(1)), Err(Error::Span));
        assert_eq!(macd(series, Some(12), Some(0), Some(1)), Err(Error::Span));
        assert_eq!(macd(series, Some(12), Some(26), Some(0)), Err(Error::Span));
    }
    assert!(macd(&empty, None, None, None).unwrap().values().is_empty());
    let single_out = macd(&single, Some(12), Some(26), Some(9)).unwrap();
    assert!(single_out.values()[0] == 0.0 && single_out.values()[0].is_sign_positive());

    println!(
        "default_last={} sig9_last={} m2_3_4_last={} leap_last={}",
        def.values()[9],
        sig9.values()[9],
        m2_3_4.values()[5],
        irreg_macd.values()[4]
    );
}
```

Standalone offline `prices` example for an external Cargo project:

```rust
use quantus_math::{prices, returns, Date, Error, ReturnKind, Series};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Daily returns -> price levels: defaults (initial=1.0, Simple) and initial=100.0 across all 3 modes
    let dates6: Vec<_> = (1..=6).map(|day| d(2019, 1, day)).collect();
    let r_simple =
        Series::try_new(dates6.clone(), vec![f64::NAN, 0.01, 0.02, -0.02, 0.0, 0.02]).unwrap();
    let p_def = prices(&r_simple, None, None).unwrap();
    let p_100 = prices(&r_simple, Some(100.0), Some(ReturnKind::Simple)).unwrap();
    let p_log = prices(&r_simple, Some(100.0), Some(ReturnKind::Logarithmic)).unwrap();
    let r_abs = Series::try_new(dates6.clone(), vec![f64::NAN, 1.0, 2.0, -2.0, 0.0, 2.0]).unwrap();
    let p_abs = prices(&r_abs, Some(100.0), Some(ReturnKind::Absolute)).unwrap();
    assert_eq!(p_def.dates(), dates6.as_slice());
    assert!((p_def.values()[5] - 1.02978792).abs() < 1e-12);
    assert!((p_100.values()[5] - 102.978792).abs() < 1e-12);
    assert!((p_log.values()[5] - 103.04545339535167).abs() < 1e-12);
    assert_eq!(p_abs.values(), &[100.0, 101.0, 103.0, 101.0, 101.0, 103.0]);

    // 2. Round-trip returns -> prices across Simple, Logarithmic, Absolute on irregular dates with leap day 2020-02-29
    let leap_dates = vec![
        d(2020, 2, 27),
        d(2020, 2, 28),
        d(2020, 2, 29),
        d(2020, 3, 2),
        d(2020, 3, 10),
    ];
    let orig = Series::try_new(leap_dates.clone(), vec![100.0, 105.0, 102.9, 108.0, 99.5]).unwrap();
    for kind in [
        ReturnKind::Simple,
        ReturnKind::Logarithmic,
        ReturnKind::Absolute,
    ] {
        let r = returns(&orig, 1, kind).unwrap();
        let restored = prices(&r, Some(orig.values()[0]), Some(kind)).unwrap();
        assert_eq!(restored.dates(), leap_dates.as_slice());
        for (&a, &e) in restored.values().iter().zip(orig.values()) {
            assert!((a - e).abs() <= 1e-12 + 1e-10 * e.abs());
        }
    }

    // 3. Missing values (leading/mid/all NaN), ±Inf, -1.0 collapse & -0.0, initial bounds, log underflow & cumprod overflow
    let d5: Vec<_> = (1..=5).map(|day| d(2020, 1, day)).collect();
    let d4 = d5[..4].to_vec();
    let d3 = d5[..3].to_vec();
    let d2 = d5[..2].to_vec();
    let lead_nan =
        Series::try_new(d5.clone(), vec![f64::NAN, f64::NAN, 0.1, f64::NAN, -0.2]).unwrap();
    let all_nan = Series::try_new(d4.clone(), vec![f64::NAN; 4]).unwrap();
    let collapse = Series::try_new(d4.clone(), vec![0.1, -1.0, 0.5, -2.0]).unwrap();
    let infs = Series::try_new(d4, vec![0.1, f64::INFINITY, 0.2, f64::NEG_INFINITY]).unwrap();
    let abs_nz = Series::try_new(d3.clone(), vec![-0.0, -0.0, 0.0]).unwrap();
    let log_under = Series::try_new(d3, vec![0.1, -800.0, 0.2]).unwrap();
    let log_cumprod = Series::try_new(d2, vec![709.0, 709.0]).unwrap();

    for (series, initial, kind) in [
        (&lead_nan, Some(100.0), Some(ReturnKind::Simple)),
        (&all_nan, Some(10.0), Some(ReturnKind::Simple)),
        (&collapse, Some(100.0), Some(ReturnKind::Simple)),
        (&infs, Some(10.0), Some(ReturnKind::Logarithmic)),
        (&abs_nz, Some(-0.0), Some(ReturnKind::Absolute)),
        (&collapse, Some(0.0), Some(ReturnKind::Simple)),
        (&collapse, Some(-0.0), Some(ReturnKind::Simple)),
        (&collapse, Some(-10.0), Some(ReturnKind::Simple)),
        (&collapse, Some(f64::NAN), Some(ReturnKind::Simple)),
        (&collapse, Some(f64::INFINITY), Some(ReturnKind::Simple)),
        (&log_under, Some(10.0), Some(ReturnKind::Logarithmic)),
        (&log_cumprod, Some(1.0), Some(ReturnKind::Logarithmic)),
    ] {
        let run1 = prices(series, initial, kind).unwrap();
        let run2 = prices(series, initial, kind).unwrap();
        assert_eq!(run1.dates(), run2.dates());
        assert!(run1
            .values()
            .iter()
            .zip(run2.values())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let col = prices(&collapse, Some(100.0), Some(ReturnKind::Simple)).unwrap();
    assert!(col.values()[1] == 0.0 && col.values()[1].is_sign_positive());
    assert!(col.values()[3] == 0.0 && col.values()[3].is_sign_negative());
    let lu = prices(&log_under, Some(10.0), Some(ReturnKind::Logarithmic)).unwrap();
    assert!(lu.values()[1] == 0.0 && lu.values()[1].is_sign_positive());
    let lc = prices(&log_cumprod, Some(1.0), Some(ReturnKind::Logarithmic)).unwrap();
    assert!(lc.values()[1].is_infinite() && lc.values()[1].is_sign_positive());

    // 4. Empty input returns empty; Logarithmic finite exp overflow returns Err(Error::Overflow)
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(prices(&empty, None, None).unwrap().values().is_empty());
    assert!(prices(&empty, Some(f64::NAN), Some(ReturnKind::Logarithmic))
        .unwrap()
        .values()
        .is_empty());
    let ovf = Series::try_new(d5[..3].to_vec(), vec![0.1, 800.0, 0.2]).unwrap();
    assert_eq!(
        prices(&ovf, Some(10.0), Some(ReturnKind::Logarithmic)),
        Err(Error::Overflow)
    );

    println!(
        "simple_def={} simple_100={} log_100={} abs_100={}",
        p_def.values()[5],
        p_100.values()[5],
        p_log.values()[5],
        p_abs.values()[5]
    );
}
```

Parity: `python3 parity/check.py --replay`, `python3 parity/check-volatility.py --replay`, `python3 parity/check-correlation.py --replay`, `python3 parity/check-beta.py --replay`, `python3 parity/check-drawdown.py --replay`, `python3 parity/check-ema.py --replay`, `python3 parity/check-smma.py --replay`, `python3 parity/check-rsi.py --replay`, `python3 parity/check-macd.py --replay`, and `python3 parity/check-prices.py --replay` build the offline replay binary and check frozen unrounded Python outputs. [The assertion map](parity/MAP.md) records the covered cases and exclusions. To regenerate vectors, use the pinned Python environment and an independently verified checkout of the upstream SHA:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-volatility.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-correlation.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-beta.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-drawdown.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-ema.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-smma.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-rsi.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-macd.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-prices.py /path/to/gs-quant
```

Generation imports the upstream Python implementation; replay does not. Compare a regenerated file and its hash before accepting any change. Upstream LICENSE, NOTICE and NOTICE.txt are preserved; source attribution is in `src/lib.rs`.
