비공식 Rust 포팅 — 원본 goldmansachs/gs-quant, Goldman Sachs 와 무관

Unofficial Rust port of goldmansachs/gs-quant; not affiliated with, endorsed or sponsored by Goldman Sachs.

Quantus Analytics is a unified repository for financial calculations. Its first package, `quantus-math`, provides offline, dependency-free moving averages, exponential moving averages, smoothed moving averages, relative strength index, moving average convergence divergence (MACD), rolling standard deviation (`std`), rolling sample covariance (`cov`), rolling linear regression (`RollingLinearRegression`), Bollinger bands (`bollinger_bands`), returns, price level reconstruction (`prices`), realized volatility (`volatility`, `vol_swap_volatility`), correlation (`correlation`, `corr_swap_correlation`), beta, maximum drawdown, offline business-day calendar & day-count utilities (`HolidayCalendar`, `DayCountConvention`, `PaymentFrequency`, `Roll`, `has_feb_29`, `day_count_fraction`, `is_business_day`, `business_day_offset`, `business_day_count`), offline multi-asset basket backtesting (`RebalFreq`, `BasketParam`, `BasketResult`, `backtest_basket`, `basket_series`), offline event-driven predefined-asset backtesting (`Timestamp`, `PredefinedOrderKind`, `PredefinedOrder`, `SimulatedFill`, `TradeStatus`, `TradeLedgerEntry`, `DailyPositionSnapshot`, `PredefinedBacktestResult`, `PredefinedBacktestInput`, `run_predefined_backtest`), and offline bucketed risk aggregation, subtraction, sorting, and market-point tenor sort order (`RiskRow`, `RiskTable`, `point_sort_order`, `sort_risk`, `aggregate_risk`, `subtract_risk`, `Error::UnitMismatch`) as a slice of [gs-quant](https://github.com/goldmansachs/gs-quant), based on upstream commit `70220e300524d54bd273be0f4c485d53b3780482`. Prices (including reconstructed levels from `prices`), moving averages, exponential moving averages, smoothed moving averages, MACD, standard deviation (`std`), Bollinger bands (`bollinger_bands`), rolling regression `fitted_values` and `standard_deviation_of_errors`, and absolute returns use the caller's input units; bucketed risk tables (`RiskTable`) carry an optional explicit unit/currency label (e.g. `Some("USD")`) validated across aggregated or subtracted tables while `point_sort_order` returns approximate calendar days from an explicit `ref_date`; basket NAV (`backtest_basket`, `basket_series`) is an excess-return index rebased to `100.0` on the first common date while realized `actual_weights` are dimensionless ratios; event-driven predefined-asset backtest NAV (`run_predefined_backtest`) starts at caller-supplied `initial_value` in cash units while EOD `weights` are dimensionless ratios and `trade_ledger` `trade_pnl` is the signed unit price spread; sample covariance (`cov`) uses the product of `x` and `y` input units; simple returns, logarithmic returns, correlation (`correlation`, `corr_swap_correlation`), beta, rolling regression `r_squared`, maximum drawdown, and day-count fractions (`day_count_fraction`) are dimensionless ratios (not percent; e.g. `-0.2` for a 20% drawdown); relative strength index (RSI) is returned on a `0`–`100` scale; volatility (`volatility`, `vol_swap_volatility`) is returned in percent (e.g. `20.0` for 20%). No network, market data, or internal engine is involved.

| Feature | Supported subset |
|---|---|
| Index | Strictly increasing unique Gregorian dates, years 1–9999; no times or time zones. Length mismatch and duplicate or unsorted dates are errors. |
| Moving average | Expanding default (`Window::Expanding`), observation window with observation ramp, or calendar-day window with calendar-day ramp. Days use `(date - size, date]`; ramp keeps dates at or after first date + ramp. NaN values are ignored; a window with no non-NaN values gives NaN. |
| Exponential moving average | Reuses `Series`, `Date`, and `Error::Beta`. `exponential_moving_average(x, beta: Option<f64>)` uses upstream default `beta = 0.75` when `None`, and decays over observation (row) order rather than calendar-day spacing, matching upstream `x.ewm(alpha=1 - beta, adjust=False).mean()`. Leading `NaN` and `±Inf` values emit `NaN` until the first finite observation; subsequent `NaN` and `±Inf` values are treated as missing, carrying forward the prior average while decaying its weight across missing rows when the next finite observation arrives. Equal finite values preserve the prior value and sign (including `-0.0 == +0.0`, even at `beta = 0.0`). Valid weights satisfy `0.0 <= beta < 1.0` (`-0.0` is accepted as `0.0`); negative values, values `>= 1.0`, `±Inf`, and `NaN` return `Err(Error::Beta)` even on empty input. |
| Smoothed moving average | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `Date`, ramp, and `Error`. `smoothed_moving_average(x, window)` mirrors upstream `gs_quant/timeseries/technicals.py:smoothed_moving_average` on a `datetime.date` index: initializes the first post-ramp value from `apply_ramp(mean(x, Window(size, 0)), w).iloc[0]` (skipping `NaN` and `±Inf` in that initial mean window, matching pandas `rolling().mean()`), slices `x` by the ramp before iterating, and applies `out[i] = ((n - 1) * out[i - 1] + x[i]) / n`. For `Window::Expanding` (`n = x.dates().len()`, ramp `0`) and `Window::Observations`, `n` is the normalized window size; for `Window::Days`, `n` is the count of post-ramp observations in `(date - size, date]`. Note that upstream calendar-day windows compare `(x.index[i] - window_size).date()`, which raises `TypeError` on a pandas `DatetimeIndex` and works only on a `datetime.date` index; Rust's date-only `Series` matches the `datetime.date` index behavior. As with `max_drawdown`, calendar-day parity with Python holds only within the pinned pandas `DateOffset` date range (whereas Rust evaluates integer day-count differences between `Date` ordinals across years 1–9999 without constructing pandas timestamps or `DateOffset` values). In the recurrence loop, `NaN` and `±Inf` propagate via IEEE-754 arithmetic (opposite-sign `±Inf` or `0.0 * ±Inf` yields `NaN`), and signed zeros follow IEEE-754 arithmetic (`[-0.0, 0.0, -0.0, -0.0]` under `Window::Observations { size: 2, ramp: 0 }` yields `[-0.0, +0.0, +0.0, +0.0]`). Empty input returns empty for `Expanding`, `Observations`, and `Days { ramp: 0 }`, and returns `Err(Error::Ramp)` for `Days { ramp > 0 }` (mapping upstream's `IndexError`). On non-empty input, `size == 0` returns `Err(Error::Window)` (including `Window::Days { size: 0, .. }`, whereas upstream `dt.date` divides by zero), observation `ramp > len` returns `Err(Error::Ramp)`, and oversize `size > len` (with valid ramp) or a ramp leaving no rows returns empty. |
| Relative strength index | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `smoothed_moving_average`, `Date`, ramp, and `Error`. `relative_strength_index(x, window: Option<Window>)` defaults to `Window::Observations { size: 14, ramp: 14 }` when `None` and follows the exact order of upstream `gs_quant/timeseries/technicals.py:relative_strength_index` on a `datetime.date` index: normalizes `window` against original `x` first (`Window::Expanding` becomes `size = x.dates().len(), ramp = 0`, which always yields an empty `Series` because `gains`/`losses` have length `x.dates().len().saturating_sub(1)`), splits one-period differences `diff(x, 1)[1:]` into non-negative `gains` and `losses`, calls `smoothed_moving_average` on each, and evaluates `100.0` when `avg_losses == 0.0` or `100.0 - 100.0 / (1.0 + avg_gains / avg_losses)` otherwise. Under this upstream rule, constant prices (`gains == 0`, `losses == 0`) take the `avg_losses == 0.0` branch and return **`100.0`** (monotonically increasing prices also return `100.0`; monotonically decreasing prices return `0.0`). Because upstream delegates calendar-day windows to `smoothed_moving_average`, which compares `(x.index[i] - window_size).date()`, upstream raises `TypeError` on a pandas `DatetimeIndex` and works only on a `datetime.date` index; Rust's date-only `Series` matches the `datetime.date` index behavior. As with `max_drawdown` and `smoothed_moving_average`, calendar-day parity with Python holds only within the pinned pandas `DateOffset` date range (whereas Rust evaluates integer day-count differences between `Date` ordinals across years 1–9999 without constructing pandas timestamps or `DateOffset` values). Mid-series `NaN` and `±Inf` propagate through `diff` and `smoothed_moving_average` (e.g. a `+Inf` jump with `avg_losses == 0.0` emits `100.0` on that row and `NaN` once `Inf / Inf` occurs). Empty input returns empty for `None`, `Expanding`, `Observations`, and `Days { ramp: 0 }`, and returns `Err(Error::Ramp)` for `Days { ramp > 0 }` (mapping upstream's `IndexError`). On non-empty input, `size == 0` returns `Err(Error::Window)` (including `Window::Days { size: 0, .. }`, whereas upstream `dt.date` divides by zero), observation `ramp > x.dates().len()` or `ramp > len(gains)` (when `gains` is non-empty) returns `Err(Error::Ramp)`, length-1 input returns empty for valid windows with `ramp <= 1` (or `Days { ramp: 0 }`) and `Err(Error::Ramp)` for `Days { ramp > 0 }` or observation `ramp > 1`, and length-2 input with `Window::Observations { size: 1, ramp: 1 }` returns empty. |
| MACD | Reuses `Series`, `Date`, and `Error::Span`. `macd(x, m: Option<usize>, n: Option<usize>, s: Option<usize>)` defaults to `m = 12`, `n = 26`, `s = 1` when `None`, and mirrors upstream `gs_quant/timeseries/technicals.py:macd` (lines 299–344) and `algebra.py:subtract` (`Interpolate.STEP` = `ffill().bfill()`). All three EWM stages decay over observation (row) order rather than calendar-day spacing (`adjust = false`, `com = (span - 1) / 2`, `alpha = 1 / (1 + com)`), preserving irregular dates including leap day `2020-02-29` verbatim. Span validation runs before empty-input handling: any `m == 0`, `n == 0`, or `s == 0` returns `Err(Error::Span)` even on an empty `Series`; any positive combination (including `m == n`, `m > n`, `m = 1`, `n = 1`, `s = 1`) is accepted. Because upstream calls `subtract(a, b)` with default `Interpolate.STEP` (`ffill().bfill()`), leading `NaN` or `±Inf` rows followed by a finite observation are backfilled with that first finite value in both `a` and `b`, producing `+0.0` in `diff = a_filled - b_filled` at every leading missing index before the third EWM (`span = s`) runs (so for `s > 1`, the first non-zero diff is smoothed against `+0.0`); if `x` has no finite values at all, the result is all `NaN`. If `a - b` overflows to `±Inf`, the third EWM treats that non-finite entry as missing, carrying forward the prior finite average while decaying `old_wt`. Finite equal values (including constant `-0.0` or mixed signed zeros) and finite length-1 series emit `+0.0`; empty input with valid spans returns empty. |
| Standard deviation (`std`) | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `Date`, ramp, and `Error`. `std(x, window)` computes the unbiased sample standard deviation (`ddof = 1`, `sqrt(sum((v - mean)^2) / (count - 1))`) using a two-pass per-window calculation, mirroring upstream `gs_quant/timeseries/statistics.py:std` and `rolling_std` (lines 55–70, 535–580) on a `datetime.date` index. `Window::Expanding` corresponds to upstream's default `Window(None, 0)`; `Window::Observations { size, ramp }` slices the trailing `size` rows and keeps `i >= ramp`; `Window::Days { size, ramp }` slices the half-open calendar range `(date - size, date]` (including irregular calendars crossing leap day `2020-02-29`) and keeps dates at or after `first_date + ramp`. Missing / non-finite handling follows upstream's two code paths: `Window::Expanding` and `Window::Observations` aggregate finite values only (`v.is_finite()`, skipping both `NaN` and `±Inf`, matching pandas `rolling().std()`), whereas `Window::Days` aggregates non-`NaN` values (`!v.is_nan()`, matching NumPy `np.std(..., ddof=1)`), so any `±Inf` inside a calendar-day window yields `NaN`. Windows with `count < 2` (including size-1 windows, singletons, index `0`, and all-`NaN` windows) emit `NaN`; constant finite windows (including `4.2`, `0.1`, and `-0.0`) emit exact `+0.0`. Empty input returns empty across all windows before window/ramp validation. On non-empty input, `size == 0` returns `Err(Error::Window)` consistently for both `Window::Observations` and `Window::Days` (whereas upstream `DateOffset(days=0)` silently falls through `rolling_std` as an expanding window), observation `ramp > len` returns `Err(Error::Ramp)`, and oversize observation `size > len` or a ramp leaving no rows (`ramp == len` or calendar ramp beyond span) returns empty. Note on large-variance overflow: when `(v - mean)^2` overflows `f64::MAX` (e.g. `1e200` alternating signs), Rust's per-window two-pass emits `+Inf` cleanly for both `Observations` and `Days` (matching upstream NumPy `rolling_std` on `Window::Days`), whereas upstream pandas observation-window rolling Welford can emit `NaN` on subsequent windows after overflow. |
| Covariance (`cov`) | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `Date`, ramp, and `Error` without adding new `Error` variants. `cov(x, y, window)` computes unbiased rolling or expanding sample covariance (`ddof = 1`) mirroring upstream `gs_quant/timeseries/statistics.py:cov` (lines 678–724) and `helper.py:{normalize_window,apply_ramp}` on a `datetime.date` index. On non-empty `x`, `size == 0` returns `Err(Error::Window)` (including `Window::Days { size: 0, .. }`, whereas upstream `DateOffset(days=0)` emits all `NaN`) and observation `ramp > x.dates().len()` returns `Err(Error::Ramp)`. Under `Window::Days { size, ramp }`, empty `x` returns an empty `Series` (whereas upstream `apply_ramp` raises `IndexError` when `ramp > 0`); non-empty `x` emits `x.dates()` at or after `first_date + ramp` and evaluates `(date - size, date]` (including leap day `2020-02-29`) over dates present in both `x` and `y` with `!vx.is_nan() && !vy.is_nan()`, dropping `±Inf` paired with `NaN` or an unmatched date before covariance while yielding `NaN` if any matched non-`NaN` pair contains `±Inf`. Under `Window::Expanding` (effective `win_size = x.dates().len(), ramp = 0`) and `Window::Observations { size, ramp }`, forms the sorted date union of `x` and `y` (`NaN` for unmatched dates): empty union returns empty; `win_size == 0` (empty `x` + non-empty `y` with `Expanding` or `size == 0`) returns `Err(Error::Window)`; `ramp > union.len()` returns `Err(Error::Ramp)`; `win_size > union.len()` or `ramp >= union.len()` returns empty; otherwise emits `union_dates[ramp..]` aggregating finite pairs (`vx.is_finite() && vy.is_finite()`, skipping both `NaN` and `±Inf`). Windows with `< 2` kept pairs emit `NaN`; constant finite `x` or constant finite `y` windows emit exact `+0.0` (eliminating upstream pandas/NumPy rounding residue such as `-2.66e-15` on `4.2` or `2.89e-34` on `0.1`); otherwise computes two-pass centered `sum((vx - mean_x) * (vy - mean_y)) / (count - 1)` with an overflow-safe `sum(v / count)` mean fallback (avoiding pandas 2.3.3 single-pass catastrophic cancellation on large-offset windows such as `1e9 + [1, 2, 4, 7, 11]`). Both `x` and `y` enforce strictly increasing unique dates via `Series::try_new` (upstream only asserts monotonicity on `x`). |
| Rolling linear regression (`RollingLinearRegression`) | Reuses `Series`, `Date`, and `Error` without adding dependencies or new `Error` variants. `RollingLinearRegression::fit(predictors: &[Series], y: &Series, window: usize, fit_intercept: bool) -> Result<Self, Error>` mirrors upstream `gs_quant/timeseries/statistics.py:RollingLinearRegression` (lines 1173–1257) on a `datetime.date` index, solving each window with pure-Rust Householder QR decomposition with column pivoting. Empty `predictors` returns `Err(Error::Length)` (vs upstream `ValueError: No objects to concatenate`). Outer-joins `predictors` onto their sorted date union (filling missing dates with `NaN`); when `fit_intercept` is `true`, prepends a constant `1.0` column unless a predictor column is already globally non-zero constant without `NaN`/`±Inf` (matching `sm.add_constant(..., has_constant='skip')`), indexing columns `0..k_cols`; when `fit_intercept` is `false`, indexes columns `1..=k_cols`. Validates `window > k_cols` before alignment (`Err(Error::Window)`), drops rows containing `NaN` or `±Inf` in any predictor or `y`, inner-joins with `y`, and requires `aligned.len() >= window` (`Err(Error::Window)` vs upstream `IndexError`/`ValueError`). Emits `NaN` on the first `window - 1` aligned dates and on rank-deficient windows (`|R[j][j]| <= max(window, k_cols) * eps * |R[0][0]|`). Uses centered `SST` for `r_squared` when any aligned design column is globally non-zero constant and uncentered `SST` otherwise. Accessors: `column_indices()`, `coefficient(i) -> Option<&Series>`, `r_squared()`, `fitted_values()`, and `standard_deviation_of_errors()`. Documented parity boundaries: (1) on constant `y` with intercept (`SST == 0`), Rust returns exact `r_squared = NaN` and `std_err = +0.0`, whereas pinned `statsmodels 0.15.0` normal-equation residual noise (`SSE ~ 1e-29`) can yield `r_squared = -Inf` and `std_err ~ 1e-15..1e-14`; (2) on ill-conditioned predictors (e.g. `1e7 + [1..6]` with intercept), Householder QR avoids squaring the condition number and recovers accurate coefficients, `r_squared`, and `std_err`, whereas `statsmodels 0.15.0` normal equations `(X'X)^-1 X'y` lose ~5 digits; (3) on exact rank-deficient (collinear) windows (e.g. `[1, x_quad, 5]` or `[1, x_quad, 2 * x_quad + 3]`), Rust Householder QR with column pivoting consistently detects rank deficiency and emits `NaN` across every singular window and output, whereas pinned `statsmodels 0.15.0` incremental `X'X` normal-equation updates can leak floating-point roundoff and emit a spurious finite solve (e.g. `coef[0] = 16.0`, `r_squared = -1.24295` on row 4 of `const-no-skip-diff-outside-y`, or `coef = [1.75, -9.953125, 4.7578125]`, `r_squared = -0.82570` on row 3 of `rank-def-collinear-all`); (4) short/empty aligned series return `Err(Error::Window)`; (5) empty `predictors` returns `Err(Error::Length)`; (6) `Series::try_new` enforces strictly increasing dates on all input series. |
| Bollinger bands (`bollinger_bands`) | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `std`, `Date`, ramp, and `Error` without adding new `Error` variants or altering `moving_average`. `bollinger_bands(x, window, k: Option<f64>) -> Result<BollingerBands, Error>` returns `BollingerBands { lower: Series, upper: Series }` with identical dates, mirroring upstream `gs_quant/timeseries/technicals.py:bollinger_bands` (lines 98–143) on a `datetime.date` index: `lower[i] = avg[i] - k * sigma[i]` and `upper[i] = avg[i] + k * sigma[i]` with default `k = 2.0` when `None`. Empty input returns empty `BollingerBands` across all windows before window or ramp validation. On non-empty input, `std(x, window)?` enforces `size == 0` → `Err(Error::Window)` consistently for both `Window::Observations` and `Window::Days` (whereas upstream `DateOffset(days=0)` falls through `rolling_std` as an expanding window), observation `ramp > len` → `Err(Error::Ramp)`, and oversize observation `size > len` or a ramp leaving no rows → empty `BollingerBands`. Missing / non-finite handling matches upstream's observation vs calendar-day split: `Window::Expanding` and `Window::Observations` skip both `NaN` and `±Inf` (`v.is_finite()`) in both `avg` and `sigma`, whereas `Window::Days` aggregates non-`NaN` values (`!v.is_nan()`) so any `±Inf` in a calendar-day window yields `NaN` bands via `sigma`. Initializing the internal mean accumulator to `-0.0` preserves upstream's signed zeros on all-`-0.0` constant windows (`lower = -0.0`, `upper = +0.0` for finite `k > 0.0` or `+0.0` (`k.is_sign_positive()`), and swapped to `lower = +0.0`, `upper = -0.0` for finite `k < 0.0` or `-0.0` (`k.is_sign_negative()`)); on mixed signed zeros (`[+0.0, -0.0]`) under `Window::Expanding` and `Window::Observations`, Rust follows IEEE-754 summation (`+0.0 + (-0.0) = +0.0`) and returns `(+0.0, +0.0)` (numeric difference `0`), whereas pinned pandas 2.3.3 `rolling().mean()` takes the GH#42064 equal-value fast path and returns `(-0.0, +0.0)` using the last zero's sign. Any `f64` value of `k` is accepted via IEEE-754 arithmetic without validation errors: `k = 0.0` or `-0.0` collapses finite-`sigma` bands to `avg` while keeping `NaN` where `sigma` is `NaN`; negative `k` (e.g. `-2.0`) swaps `lower` and `upper` (`lower > upper`); `k = NaN` yields all-`NaN` bands; and `k = ±Inf` yields `∓Inf` / `±Inf` where `sigma > 0.0` and `NaN` where `sigma == 0.0` or `NaN` (including constant calendar-day windows such as ten days of `4.2` under `Window::Days { size: 10, ramp: 0 }`, where Rust reuses `std`'s exact `+0.0` contract to emit `NaN`, whereas pinned NumPy 2.2.6 `rolling_std` leaves a `~9.36e-16` mean-rounding residue that amplifies to `(-Inf, +Inf)` / `(+Inf, -Inf)`). |
| Returns | Positive integer observation lag, SIMPLE, LOGARITHMIC and ABSOLUTE. NaN propagates. Logarithmic returns reject nonpositive prices. Division by zero follows floating-point arithmetic. |
| Prices | Reuses `Series`, `ReturnKind`, `Date`, and `Error::Overflow`. `prices(x, initial: Option<f64>, kind: Option<ReturnKind>)` reconstructs price levels from a returns series with defaults `initial = 1.0` and `kind = ReturnKind::Simple`, mirroring upstream `gs_quant/timeseries/econometrics.py:prices` (lines 353–421) and `statistics.py:{product,sum_}`. Round-trips `returns(px, 1, kind)` back to `px` via `prices(&r, Some(px[0]), Some(kind))` across `Simple`, `Logarithmic`, and `Absolute` while preserving irregular dates including leap day `2020-02-29`. Empty input returns empty regardless of `initial` (even `NaN`/`±Inf`) or `kind`. All modes skip `NaN` and `±Inf` (matching pandas 2.3.3 `rolling(len, 0)`): `ReturnKind::Simple` multiplies finite `1.0 + v` factors starting from `prod = 1.0` and emits `prod * initial` (so leading `NaN` emits `initial`, a `-1.0` return collapses subsequent finite products to `0.0`, and a later `< -1.0` return flips the sign to `-0.0`); `ReturnKind::Logarithmic` first returns `Err(Error::Overflow)` if any finite `v` has `v.exp().is_infinite()` (e.g. `800.0`, matching Python `math.exp` `OverflowError`), skips `+Inf` without error, maps `-Inf` and underflow (e.g. `-800.0`) to `0.0`, allows cumulative product overflow of finite exponentials (e.g. `[709.0, 709.0]` → `+Inf`), and emits `prod * initial`; `ReturnKind::Absolute` sums finite values (emitting `+0.0 + initial` before the first finite value, and accumulating from `-0.0` once a finite value appears so `-0.0` is preserved when all finite inputs and `initial` are `-0.0`). For `ReturnKind::Absolute`, parity applies to finite-sum inputs without overflow cancellation: when large finite values overflow the running sum to `±Inf` (e.g. `[1e308, 1e308, -1e308]` or `[1e308, 1e308, 1.0]`), upstream pandas `rolling().sum()` uses compensated Kahan summation where `(+Inf - +Inf)` yields `NaN`, whereas Rust uses plain `f64` summation (staying `+Inf`). |
| Volatility | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<ReturnKind>` (`Simple`, `Logarithmic`, `Absolute`, or `None` for precomputed returns). Standard sample volatility (`assume_zero_mean = false`, `n - 1` denominator, `NaN` for `< 2` non-NaN values) and zero-mean RMS (`assume_zero_mean = true`, `n` denominator, `NaN` for `0` non-NaN values). Explicit `Option<u32>` annualization factor (including `0`, which scales finite values to `0.0` and keeps `NaN`), or `None` for upstream automatic classification (`< 2.1d → 252`, `[6, 8) → 52`, `[14, 17) → 26`, `[25, 35) → 12`, `[85, 97) → 4`, `[360, 386) → 1`). Output is annualized in percent (`* sqrt(factor) * 100`). |
| Correlation | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), and `Option<(ReturnKind, ReturnKind)>` (`Some((kx, ky))` computes lag-1 returns on `x` and `y` independently before date intersection; `None` treats both inputs as precomputed returns). Aligns common dates and drops any pair with a `NaN` before rolling (`Observations` counts aligned non-NaN pairs; `Days` uses `(date - size, date]`). Reindexes results onto `x`'s dates (unmatched left dates become `NaN`) and applies the outer ramp on `x`. Returns a dimensionless ratio (not percent). Sample Pearson (`assume_zero_mean = false`) yields `NaN` when a window has `< 2` pairs or zero variance. Zero-mean mode (`assume_zero_mean = true`) computes normalized inner products via upstream's `mean` + zero-mean `volatility` pipeline on the aligned pairs: both the inner window/ramp (on aligned pairs) and outer ramp (on `x`) apply, so dates masked by the inner ramp remain `NaN` even if they survive the outer ramp; empty, singleton, irregular-spacing, or oversize-window aligned pairs (including `Window::Expanding` when aligned count `< x.len()`) return `Err(Error::AnnualizationFactor)`, and inner observation ramp `> aligned.len()` returns `Err(Error::Ramp)`. |
| Beta | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `ReturnKind::Simple` (internally when `prices = true`), `Error`, and outer ramp. When `prices = true`, computes lag-1 simple returns independently on `x` and `benchmark`; when `prices = false`, treats both series as precomputed returns. For `Window::Expanding` (size = `x.dates().len()`) and `Window::Observations`, computes rolling sample covariance (`ddof = 1`, skipping `NaN` pairs) over the outer-joined date union divided by `benchmark`'s own rolling sample variance (`ddof = 1`, skipping `NaN`s) aligned by date—without pooling the two sample sets. For `Window::Days`, computes sample covariance divided by benchmark variance over `(date - size, date]` on the date intersection, propagating any `NaN` inside the window. Masks the first 3 positions of the intermediate result (`union` for observations/expanding, `intersection` for days) to `NaN`, reindexes onto `x`'s dates (`NaN` for unmatched dates), and applies the outer ramp. An exactly constant finite benchmark window yields `NaN` (zero denominator). |
| Max drawdown | Reuses `Series`, `Window` (`Expanding`, `Observations`, `Days`), `Date`, ramp, and `Error`. Two-stage rolling peak-to-trough ratio: computes the rolling maximum price, forms `price / rolling_max - 1`, and takes the rolling minimum of those ratios over the same window before applying the ramp (e.g. `-0.2` for a 20% drawdown). `Window::Expanding` uses `x.dates().len()` with ramp `0`; `Window::Observations` skips `NaN` and `±Inf` in both rolling stages (matching upstream pandas `rolling().max()` / `rolling().min()`); `Window::Days` uses `(date - size, date]` in both stages without interpolating holidays, skipping `NaN` while retaining `±Inf` (matching upstream pandas `Series.max()` / `Series.min()`). All-NaN windows and `0 / 0` stay `NaN`. Empty input returns empty for `Expanding`, `Observations`, and `Days { ramp: 0 }`, and returns `Err(Error::Ramp)` for `Days { ramp > 0 }` (mapping upstream's `IndexError` when applying a positive calendar ramp to an empty date index). For `Window::Days`, date parity with Python holds only when the input dates, every left window boundary (`date - size` days), and the ramp cutoff (`first_date + ramp` days) are representable and computable with the pinned pandas `DateOffset`. Rust evaluates calendar windows and ramps via integer day-count differences between `Date` ordinals (without constructing pandas timestamps or `DateOffset` values), so extremes that raise Python date-range or `DateOffset` errors (`OverflowError` / `OutOfBoundsDatetime`) can still produce normal values or an empty `Series` in Rust—for example, on `2024-01-01`/`2024-01-05` with prices `[2.0, 1.0]`, an ordinary window such as `size: 100, ramp: 0` matches Python (`[0.0, -0.5]`), whereas `size: 1_000_000` returns `[0.0, -0.5]` and `ramp: 1_000_000` returns empty in Rust while Python errors; similarly, near pandas's lower timestamp limit, `1677-09-22`/`1677-09-23` with `size: 2` succeeds in Rust (`[0.0, -0.5]`) even though the left boundary `date - 2` days is out of range for pandas. |
| Numerical precision & non-finite inputs | Parity tolerance `1e-12 + 1e-10 * |expected|` applies to well-conditioned finite inputs. For `volatility`, `correlation`, and `beta`, inputs with `±Inf` are excluded from compatibility vectors: Rust follows IEEE-754 arithmetic (`NaN` for sample volatility, Pearson correlation, and beta, `+Inf` for RMS, `NaN` for zero-mean `Inf / Inf`), whereas upstream pandas `rolling()` and numpy `rolling_std()` diverge on `±Inf`. For `max_drawdown`, `±Inf` (including `-Inf` produced by a negative price following a `+0.0` peak) matches upstream: integer/expanding windows skip `±Inf` in both rolling stages while calendar-day windows retain `±Inf`, except when a calendar-day window's rolling peak ties across mixed-sign zeros (`+0.0` and `-0.0`). On mixed-sign zero peak ties in `Window::Days`, Rust keeps the last zero in the window (updating on `!(cur > v)`), so a negative price divides by that zero's sign (`-Inf` when the last zero is `+0.0`, `+Inf` when it is `-0.0`); even within the same pinned Python/NumPy/pandas environment, the resulting `±Inf` sign for mixed-sign zeros can vary depending on NumPy CPU optimizations. For constant non-power-of-two float series (e.g. `0.01`, `0.1`), Rust detects finite constant windows before centering and returns `NaN` (zero variance / zero denominator), whereas upstream pandas 2.3.3 observation-window `rolling().corr()` can emit `±Inf` from covariance rounding residue divided by zero standard deviation (e.g. on `0.01`, or `0.1` vs varying), upstream calendar-day `Series.corr` (`np.corrcoef`) can emit rounding-residue `1.0` (constant/constant) or `0.0` (constant/varying) when `sum(x) / n != x` (e.g. three `0.1` values), and upstream calendar-day `beta` (`np.cov` / `np.var`) emits `±21.333333333333332` (`±64/3`) on constant `0.1` over a 6-day window (`constant-calendar` vector) due to floating-point rounding residue in the denominator variance. For inputs with very large constant offsets relative to spread (e.g. `1e9 + 0.1`, where `f64` input ULP is `~1.2e-7`), Rust uses two-pass centered sums (`sum((x - mean)^2)` / `sum((x - mean_x)(y - mean_y))`), avoiding catastrophic cancellation while upstream pandas observation-window and numpy day-window paths diverge by `~6e-7` relative error (outside the `1e-10` parity bound). For `exponential_moving_average`, `±Inf` inputs match upstream pandas (treated as missing like `NaN`), and signed zeros match upstream bit-for-bit; for `beta = NaN`, upstream `gs-quant` documents `0 <= beta < 1` (`technicals.py:265-266`) while pandas `get_center_of_mass` checks `alpha <= 0 or alpha > 1` and lets `NaN` slip through (producing alternating artifacts such as `[2.0, NaN, 4.0]` on `[2.0, 3.0, 4.0]`), whereas Rust strictly enforces `[0.0, 1.0)` and returns `Err(Error::Beta)`. For `prices`, `NaN` and `±Inf` skipping, `-1.0` collapse, signed zeros, `ReturnKind::Logarithmic` `Error::Overflow` on finite `exp` overflow, log underflow, and cumulative product overflow all match upstream; only extreme `ReturnKind::Absolute` inputs whose finite running sum overflows `f64` diverge from pandas's Kahan summation (`+Inf` in Rust vs `NaN` in pandas). For `std`, `Window::Expanding` and `Window::Observations` skip `NaN` and `±Inf` while `Window::Days` skips `NaN` and returns `NaN` when `±Inf` is present; constant finite windows (including `0.1` and `-0.0`) return `+0.0` (within `<= 1.7e-17` of NumPy's `rolling_std` on `0.1`); and in extreme variance-overflow regimes (`1e200`), Rust's per-window two-pass returns `+Inf` cleanly whereas pandas observation-window rolling Welford can poison subsequent windows to `NaN` after overflow. For `bollinger_bands`, `avg` and `sigma` match upstream on both observation/expanding (`v.is_finite()`, skipping `NaN` and `±Inf`) and calendar-day (`!v.is_nan()`, `±Inf` → `NaN`) paths, all-`-0.0` constant windows preserve `lower = -0.0` and `upper = +0.0` for finite `k > 0.0` or `+0.0` (`k.is_sign_positive()`, swapping to `(+0.0, -0.0)` for finite `k < 0.0` or `-0.0`), and `k` edge values (`0.0`, `-0.0`, negative, `NaN`, `±Inf`) follow IEEE-754 arithmetic without validation errors. Two upstream signed-zero / rounding-residue differences are documented and tracked in `parity/bollinger-bands-vectors.json`: (1) on mixed signed zeros (`[+0.0, -0.0]`) under `Window::Expanding` and `Window::Observations`, Rust follows IEEE-754 `+0.0 + (-0.0) = +0.0` and emits `(+0.0, +0.0)` (numeric difference `0`), whereas pinned pandas 2.3.3 `rolling().mean()` uses its GH#42064 equal-value fast path and adopts the last zero's sign `-0.0` to emit `(-0.0, +0.0)`; and (2) on constant calendar-day windows such as ten days of `4.2` (`Window::Days { size: 10, ramp: 0 }`) with `k = ±Inf`, Rust preserves `std`'s exact `+0.0` constant-window contract so `0.0 * ±Inf = NaN`, whereas pinned NumPy 2.2.6 `rolling_std` retains a `~9.36e-16` mean-rounding residue that multiplies with `±Inf` into `(-Inf, +Inf)` / `(+Inf, -Inf)`. For `cov`, `Window::Expanding` and `Window::Observations` skip both `NaN` and `±Inf` (`vx.is_finite() && vy.is_finite()`) while `Window::Days` drops `NaN` pairs and unmatched dates first (`!vx.is_nan() && !vy.is_nan()`) and emits `NaN` if any remaining matched pair contains `±Inf`; constant finite windows return exact `+0.0` (avoiding `-2.66e-15` in pandas `rolling().cov()` on `4.2` and `2.89e-34` in NumPy `np.cov` on `0.1`); and large-offset observation/expanding windows (e.g. `1e9 + [1, 2, 4, 7, 11]` vs `1e9 + [2, 5, 3, 9, 12]`) use two-pass centered covariance matching exact mathematics and `np.cov`, avoiding pinned pandas 2.3.3 `rolling().cov()` catastrophic cancellation (`-192.0`). |
| Calendar & day-count | Reuses `Date` (with `year()`, `month()`, `day()` accessors) and `Error::Date` without adding dependencies or new `Error` variants. Mirrors `gs_quant/datetime/date.py` and explicit-holiday `numpy.busdaycalendar` semantics: `has_feb_29(start, end)` (`start < YYYY-02-29 <= end`, `false` when `start >= end`, proleptic Gregorian century rules), `day_count_fraction(start, end, convention, frequency)` (`Actual360`, `Actual364`, `Actual36525`, `Actual365F`, `Actual365L` with `Annually` vs non-`Annually` leap-year rules, and `OneOne == 1.0`), `HolidayCalendar::try_new(week_mask, holidays)` / `HolidayCalendar::weekdays(holidays)` (Monday=0..Sunday=6 mask; sorts, deduplicates, and filters holidays to valid weekdays; all-`false` `week_mask` returns `Err(Error::Date)`), `is_business_day(date, calendar)`, `business_day_offset(date, offset, roll, calendar)` (`O(log H)` two-step `np.busday_offset` semantics across `Roll::{Raise, Forward, Backward, ModifiedFollowing, ModifiedPreceding}`), and `business_day_count(begin, end, calendar)` (`O(log H)` exact NumPy asymmetry: `+[begin, end)` for `begin <= end` and `-(end, begin]` for `begin > end`). Documented parity boundaries: (1) explicit `HolidayCalendar` is required—remote `GsCalendar` exchange/currency dataset fetches (`HOLIDAY`, `HOLIDAY_CURRENCY`) are out of scope; (2) valid `Date` range is Gregorian `0001-01-01..=9999-12-31`, returning `Err(Error::Date)` on underflow/overflow (whereas `np.datetime64[D]` supports wider 64-bit day offsets); (3) `Roll::Raise` on a non-business day and all-`false` `week_mask` return typed `Err(Error::Date)` (vs NumPy `ValueError`). |
| Multi-asset basket backtesting | Reuses `Series`, `Date`, and `Error::Length` without adding dependencies or new `Error` variants. `backtest_basket(series, weights, costs, rebal_freq)` returns `BasketResult { nav, actual_weights }` (excess-return NAV starting at `100.0` plus realized weight series in asset order), and `basket_series(series, weights, costs, rebal_freq)` returns `nav`, mirroring `gs_quant/timeseries/backtesting.py` lines 52–180. Accepts `weights` and `costs` as `Option<&[BasketParam<'_>]>` (`Fixed(f64)` or time-varying `Series(&Series)`); `None` defaults to `1/N` equal weights and `0.0` execution costs. Aligns onto the sorted common calendar intersection across all price series and all `BasketParam::Series` weight/cost inputs. Supports `RebalFreq::{Daily, Weekly, Monthly}` with upstream's strict `hyp < max(cal)` rule, holiday-gap snapping with duplicate-snap deduplication, and anchor-based month clamping (`2024-01-31 -> 2024-02-29 -> 2024-03-31 -> 2024-04-30`). Reproduces upstream IEEE-754 recurrence including signed cost deduction `nav_i -= cost_rate * nav_i` on rebalance dates, short/leveraged/zero weights, and `NaN`/zero-price/zero-NAV/negative-NAV propagation. Documented parity boundaries: (1) remote `Basket` class (`GsAssetApi`/`GsDataApi`) is intentionally out of scope; (2) `weights: None` / `costs: None` represent defaults while explicit empty slices `Some(&[])` on non-empty `series` return `Err(Error::Length)` (vs Python's falsy `or` fallback); (3) empty `series`, length mismatch, and disjoint/empty calendar intersection return typed `Err(Error::Length)` (vs Python `ZeroDivisionError`, `MqValueError`, `IndexError`); (4) duplicate/unsorted dates and `DatetimeIndex` time zones are rejected by `Series::try_new`. |
| Bucketed risk aggregation, subtraction, sorting & `point_sort_order` | Adds `RiskRow`, `RiskTable`, `point_sort_order`, `sort_risk`, `aggregate_risk`, `subtract_risk`, and a single semantic error variant `Error::UnitMismatch` in pure `std` Rust (zero external dependencies). Mirrors `gs_quant/datetime/point.py` lines 25–325 and `gs_quant/risk/core.py` lines 33–34, 516–682. `point_sort_order(point, ref_date)` implements all upstream tenor/date/const/compound `;`/copula/futures/repo patterns in exact precedence with POSIX `%y` 68/69 pivot and returns `None` where upstream leaves `days = None` (treated as `0.0` by `sort_risk`). `sort_risk(table, ref_date)` stable-sorts rows by `(date, time, mkt_type, mkt_asset, mkt_class, point_sort_order(mkt_point, ref_date).unwrap_or(0.0))` preserving input order on ties (`30d` vs `1m`). `aggregate_risk(tables, threshold, ref_date)` rejects empty `tables` (`Err(Error::Length)`), validates unit compatibility across non-empty `unit` labels (`Err(Error::UnitMismatch)`, inheriting non-empty unit over `None`/`Some("")`), replaces `NaN` with `0.0` (`fillna(0)` while propagating `±Inf`), sums buckets in canonical lexicographical order (matching pandas `groupby(..., sort=True)` before `sort_risk`), filters by strict `abs(value) > threshold` (`== threshold`, `0.0` at `threshold = 0.0`, and all rows at `NaN`/`+Inf` dropped), and sorts via `sort_risk`. `subtract_risk(left, right, ref_date)` validates unit compatibility (`Err(Error::UnitMismatch)`) and optional `date`/`time` presence compatibility on non-empty tables (`Err(Error::Length)`), negates `right`, and delegates to `aggregate_risk`. Documented parity boundaries: (1) upstream `gs_quant/risk/core.py:642-643` (`subtract_risk`) asserts `'value' in left.columns.names`, inspecting pandas `Index.names` (`[None]`) instead of column names and raising `AssertionError` on standard DataFrames—Rust implements the documented/intended semantics; (2) Rust requires an explicit `ref_date: Date` instead of reading `datetime.date.today()`, guaranteeing deterministic offline behavior; (3) only already-resolved bucketed risk tables/rows are supported—remote `PricingContext`, `Future`, server pricing, and `ErrorValue` are out of scope. |
| Volatility & correlation swaps (`vol_swap_volatility`, `corr_swap_correlation`) | Reuses `Series`, `Date`, `volatility`, `correlation`, `ReturnKind::Logarithmic`, `Window::Observations`, and `Error` without adding dependencies or new `Error` variants. Mirrors `gs_quant/timeseries/econometrics.py:{vol_swap_volatility,corr_swap_correlation}` (lines 695–765 and 891–964). `vol_swap_volatility(prices, n_days, annualization_factor, assume_zero_mean)` returns annualized volatility in percent using logarithmic returns over `Window::Observations { size: n, ramp: n - 1 }` (`n = n_days.unwrap_or(prices.dates().len())`, default `annualization_factor = 252`); empty `prices` returns empty before validating `n_days`, `n == 0` returns `Err(Error::Window)`, `n == len + 1` returns empty, `n >= len + 2` returns `Err(Error::Ramp)`, and non-positive prices return `Err(Error::NonPositivePrice)`. `corr_swap_correlation(x, y, n_days, assume_zero_mean)` returns dimensionless realized correlation in `[-1, 1]` using logarithmic returns over `Window::Observations { size: n - 1, ramp: n - 2 }` (`n = n_days.unwrap_or_else(|| x.dates().len().min(y.dates().len()))`); empty `x` returns empty before validating `n_days`, `n <= 1` returns `Err(Error::Window)`, non-positive prices return `Err(Error::NonPositivePrice)`, and zero-mean mode propagates underlying `Error::AnnualizationFactor` / `Error::Ramp` from `correlation`. Documented parity boundaries: (1) upstream's `Window(w, r)` validation (`r != w - 1` → `MqTypeError`) is enforced by construction via Rust's typed `Option<usize>` parameter; (2) constant zero-variance price/log-return series with `assume_zero_mean = false` in `corr_swap_correlation` inherit `correlation`'s `NaN` contract (vs upstream pandas `±Inf` rounding residue); (3) zero new Cargo dependencies and zero new `Error` variants. |
| Event-driven predefined-asset backtesting (`run_predefined_backtest`) | Adds `Timestamp`, `PredefinedOrderKind` (`MarketOnClose`, `Twap`, `AtMarket`, `Cost`), `PredefinedOrder`, `SimulatedFill`, `TradeStatus`, `TradeLedgerEntry`, `DailyPositionSnapshot`, `PredefinedBacktestResult`, `PredefinedBacktestInput`, and `run_predefined_backtest` in pure `std` Rust (zero new dependencies, zero new `Error` variants). Mirrors `gs_quant/backtests/{predefined_asset_engine,execution_engine,order,backtest_objects,data_handler,data_sources}.py` (flagship vector `test_backtest_predefined`). Adjusts `start`/`end` backward onto explicit `HolidayCalendar`, records initial NAV `(adjusted_start, initial_value)`, builds a sorted/deduplicated UTC wall-clock timer over active business days `(adjusted_start, adjusted_end]`, and executes the exact FIFO `VecDeque` event loop (`ping` fills → `Market` order submission → `Valuation` EOD mark-to-market with `1e-12` zero-position pruning → `Order` insertion stably sorted by `execution_end_time`). `Twap` averages intraday ticks strictly over half-open `(start, end]`, `MarketOnClose` fills at `23:00:00` daily close, `AtMarket` fills at exact intraday tick, and `Cost` applies cash-only adjustments at price `0.0`. Produces FIFO `trade_ledger` over executed orders grouped by first-seen instrument (`Cost` mapped to `cash_label`) with `trade_pnl = Some((close_price - open_price) * sign(open_qty))` (exact `np.sign`). Documented parity boundaries: (1) requires explicit `HolidayCalendar`, daily/intraday price tables, and explicit `PredefinedOrder` events — remote `GsDataSource`, `GsCalendar`, `PricingContext`, `GenericEngine`, live order routing, and IANA timezone conversions are out of scope (all timestamps are UTC wall-clock); (2) missing or `NaN` execution/valuation prices return typed `Err(Error::Length)` (vs upstream `KeyError`/`RuntimeError`), invalid date/time ranges return `Err(Error::Date)`, and unsorted/duplicate intraday timestamps return `Err(Error::UnsortedOrDuplicate)`. |
| Not supported | Remote `GsDataSource`, `GenericEngine`, live order routing, IANA timezone conversions, remote `PricingContext`, `Future`, server risk pricing, `ErrorValue`, remote `Basket` class (`GsAssetApi`/`GsDataApi`), remote `GsCalendar` exchange/currency holiday dataset fetches (`HOLIDAY`, `HOLIDAY_CURRENCY`), date-based return lag, months/weeks (`1m`, `3m`, `1w`), mixed-unit ramps, duplicate or unsorted indices, `DatetimeIndex` / time zones, non-date indices (`TypeError` in upstream `max_drawdown`), pandas nullable values distinct from NaN, non-simple return kinds for `beta`, variance swap wrappers, service-level portfolio beta (`measures_portfolios.py`) and drawdown report services, and other gs-quant functions. This is not a claim of full upstream compatibility. |

Observation window size must be positive and its ramp at most the number of rows; for `moving_average`, `std`, `bollinger_bands`, `smoothed_moving_average`, `relative_strength_index` (against `gains` of length `x.len().saturating_sub(1)` after initial normalization on `x`), Pearson `correlation`, `beta`, and `max_drawdown`, size larger than the effective series returns an empty series (or empty `BollingerBands`) after rejecting an invalid observation ramp with `Err(Error::Ramp)`. For `cov`, observation/expanding windows validate against `x` when `x` is non-empty and then against the sorted date union of `x` and `y` (so `empty x` + `non-empty y` returns `Err(Error::Window)` for `Window::Expanding` / `size == 0` and `Err(Error::Ramp)` for `ramp > y.len()`, while `size > union.len()` returns empty); under `Window::Days`, empty `x` returns empty (whereas upstream `apply_ramp` raises `IndexError` when `ramp > 0`) and non-empty `x` rejects `size == 0` with `Err(Error::Window)`. For `volatility`, size larger than the series returns empty when `annualization_factor` is `Some(...)`, and `Err(Error::AnnualizationFactor)` when `None`. For zero-mean `correlation`, size larger than the aligned non-NaN pair count (including `Window::Expanding` when `aligned.len() < x.len()`) returns `Err(Error::AnnualizationFactor)` unless an inner observation ramp `> aligned.len()` triggers `Err(Error::Ramp)` first. Calendar-day size must be positive. Empty `moving_average`, `std`, `bollinger_bands`, `volatility`, and left-input `correlation` and `beta` return empty before window or annualization validation; empty `max_drawdown`, `smoothed_moving_average`, and `relative_strength_index` return `Err(Error::Ramp)` when given a positive calendar-day ramp (`Window::Days { ramp > 0, .. }`, mapping upstream's `IndexError`) and empty otherwise; `returns` rejects lag zero even on empty input; `exponential_moving_average` validates `0.0 <= beta < 1.0` even on empty input, returning `Err(Error::Beta)` for negative values, values `>= 1.0`, `±Inf`, and `NaN`; `macd` validates `m >= 1`, `n >= 1`, `s >= 1` even on empty input, returning `Err(Error::Span)` if any span is `0`; `prices` returns empty on empty input regardless of `initial` or `kind`, and returns `Err(Error::Overflow)` in `ReturnKind::Logarithmic` when any finite `v` overflows `v.exp()`. A larger-than-series lag gives all NaNs. Singleton or unclassifiable spacing under automatic annualization returns `Err(Error::AnnualizationFactor)`. Use `Window::Observations { size: n, ramp: n }` for the upstream integer-window default ramp and `Window::Days { size: n, ramp: n }` for its day-string default. Public errors are `Error` variants.

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

Standalone offline `std` example for an external Cargo project:

```rust
use quantus_math::{std as series_std, Date, Error, Series, Window};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Daily series sample standard deviation (ddof = 1): Expanding, Observations, and 7d Days
    let dates = vec![
        d(2019, 1, 1),
        d(2019, 1, 2),
        d(2019, 1, 3),
        d(2019, 1, 4),
        d(2019, 1, 7),
        d(2019, 1, 8),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let exp = series_std(&x, Window::Expanding).unwrap();
    let obs2 = series_std(&x, Window::Observations { size: 2, ramp: 0 }).unwrap();
    let cal7d = series_std(&x, Window::Days { size: 7, ramp: 0 }).unwrap();
    assert!(exp.values()[0].is_nan());
    assert!((exp.values()[5] - 1.6733200530681511).abs() < 1e-12);
    assert!((obs2.values()[5] - 2.1213203435596424).abs() < 1e-12);
    assert!((cal7d.values()[5] - 1.8708286933869707).abs() < 1e-12);

    // 2. Irregular calendar crossing leap day 2020-02-29 with Days { 3, 3 } and Days { 7, 0 }
    let leap_dates = vec![
        d(2020, 2, 27),
        d(2020, 2, 28),
        d(2020, 2, 29),
        d(2020, 3, 2),
        d(2020, 3, 5),
    ];
    let leap = Series::try_new(leap_dates.clone(), vec![1.0, 3.0, 2.0, 6.0, 10.0]).unwrap();
    let d3_r3 = series_std(&leap, Window::Days { size: 3, ramp: 3 }).unwrap();
    let d7_r0 = series_std(&leap, Window::Days { size: 7, ramp: 0 }).unwrap();
    assert_eq!(d3_r3.dates(), &[d(2020, 3, 2), d(2020, 3, 5)]);
    assert!((d3_r3.values()[0] - 2.8284271247461903).abs() < 1e-12);
    assert!(d3_r3.values()[1].is_nan());
    assert!((d7_r0.values()[4] - 3.593976442141304).abs() < 1e-12);

    // 3. Constants (4.2, 0.1, -0.0), NaN, ±Inf (obs skips vs days -> NaN), size-1/singleton/oversize/ramp==len, determinism
    let c42 = Series::try_new(dates.clone(), vec![4.2; 6]).unwrap();
    let c01 = Series::try_new(dates.clone(), vec![0.1; 6]).unwrap();
    let cneg0 = Series::try_new(dates.clone(), vec![-0.0; 6]).unwrap();
    let mid_nan =
        Series::try_new(dates.clone(), vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, 5.0]).unwrap();
    let all_nan = Series::try_new(dates.clone(), vec![f64::NAN; 6]).unwrap();
    let infs = Series::try_new(
        dates.clone(),
        vec![1.0, f64::INFINITY, 3.0, 5.0, f64::NEG_INFINITY, 7.0],
    )
    .unwrap();
    let single = Series::try_new(vec![d(2020, 1, 1)], vec![5.0]).unwrap();

    for (series, window) in [
        (&c42, Window::Observations { size: 3, ramp: 0 }),
        (&c01, Window::Days { size: 3, ramp: 0 }),
        (&cneg0, Window::Observations { size: 3, ramp: 0 }),
        (&mid_nan, Window::Observations { size: 3, ramp: 0 }),
        (&all_nan, Window::Days { size: 3, ramp: 0 }),
        (&infs, Window::Observations { size: 3, ramp: 0 }),
        (&infs, Window::Days { size: 4, ramp: 0 }),
        (&x, Window::Observations { size: 1, ramp: 0 }),
        (&single, Window::Expanding),
        (&x, Window::Observations { size: 10, ramp: 0 }),
        (&x, Window::Observations { size: 2, ramp: 6 }),
    ] {
        let run1 = series_std(series, window).unwrap();
        let run2 = series_std(series, window).unwrap();
        assert_eq!(run1.dates(), run2.dates());
        assert!(run1
            .values()
            .iter()
            .zip(run2.values())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let inf_obs = series_std(&infs, Window::Observations { size: 3, ramp: 0 }).unwrap();
    let inf_days = series_std(&infs, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert!((inf_obs.values()[2] - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!(inf_days.values().iter().all(|v| v.is_nan()));
    let c01_out = series_std(&c01, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert!(c01_out.values()[2] == 0.0 && c01_out.values()[2].is_sign_positive());

    // 4. Empty -> empty across all windows; non-empty size == 0 -> Error::Window, ramp > len -> Error::Ramp
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 0, ramp: 0 },
        Window::Days { size: 2, ramp: 2 },
    ] {
        assert!(series_std(&empty, w).unwrap().values().is_empty());
    }
    assert_eq!(
        series_std(&x, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        series_std(&x, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        series_std(&x, Window::Observations { size: 2, ramp: 7 }),
        Err(Error::Ramp)
    );

    println!(
        "exp={} obs2={} cal7d={} leap_3d_r3d={} leap_7d={}",
        exp.values()[5],
        obs2.values()[5],
        cal7d.values()[5],
        d3_r3.values()[0],
        d7_r0.values()[4]
    );
}
```

Standalone offline `bollinger_bands` example for an external Cargo project:

```rust
use quantus_math::{bollinger_bands, BollingerBands, Date, Error, Series, Window};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Daily series Bollinger bands: default k = None (2.0) and custom k = Some(1.5)
    let dates: Vec<_> = (1..=6).map(|day| d(2019, 1, day)).collect();
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let exp_def: BollingerBands = bollinger_bands(&x, Window::Expanding, None).unwrap();
    let obs_k15 =
        bollinger_bands(&x, Window::Observations { size: 3, ramp: 0 }, Some(1.5)).unwrap();
    let cal_2d = bollinger_bands(&x, Window::Days { size: 2, ramp: 2 }, None).unwrap();
    assert!(exp_def.lower.values()[0].is_nan() && exp_def.upper.values()[0].is_nan());
    assert!((exp_def.lower.values()[5] - (-0.34664010613630225)).abs() < 1e-12);
    assert!((exp_def.upper.values()[5] - 6.346640106136302).abs() < 1e-12);
    assert!((obs_k15.lower.values()[5] - (-0.4415838843020419)).abs() < 1e-12);
    assert!((obs_k15.upper.values()[5] - 7.108250550968709).abs() < 1e-12);
    assert_eq!(cal_2d.lower.dates(), &dates[2..]);

    // 2. Irregular calendar crossing leap day 2020-02-29 with Days { 3, 0 } and Days { 3, 3 }
    let leap_dates = vec![
        d(2020, 2, 27),
        d(2020, 2, 28),
        d(2020, 2, 29),
        d(2020, 3, 2),
        d(2020, 3, 5),
    ];
    let leap = Series::try_new(leap_dates.clone(), vec![10.0, 12.0, 11.0, 15.0, 20.0]).unwrap();
    let d3_r0 = bollinger_bands(&leap, Window::Days { size: 3, ramp: 0 }, None).unwrap();
    let d3_r3 = bollinger_bands(&leap, Window::Days { size: 3, ramp: 3 }, None).unwrap();
    assert_eq!(d3_r0.lower.dates(), leap_dates.as_slice());
    assert!((d3_r0.lower.values()[2] - 9.0).abs() < 1e-12);
    assert!((d3_r0.upper.values()[2] - 13.0).abs() < 1e-12);
    assert_eq!(d3_r3.lower.dates(), &[d(2020, 3, 2), d(2020, 3, 5)]);
    assert!((d3_r3.lower.values()[0] - 7.343145750507619).abs() < 1e-12);
    assert!((d3_r3.upper.values()[0] - 18.65685424949238).abs() < 1e-12);
    assert!(d3_r3.lower.values()[1].is_nan() && d3_r3.upper.values()[1].is_nan());

    // 3. Constants (4.2, -0.0), NaN, ±Inf (obs skips vs days -> NaN), k edges, determinism
    let c42 = Series::try_new(dates.clone(), vec![4.2; 6]).unwrap();
    let cneg0 = Series::try_new(dates.clone(), vec![-0.0; 6]).unwrap();
    let mid_nan =
        Series::try_new(dates.clone(), vec![3.0, f64::NAN, 5.0, f64::NAN, f64::NAN, 7.0]).unwrap();
    let all_nan = Series::try_new(dates.clone(), vec![f64::NAN; 6]).unwrap();
    let infs = Series::try_new(
        dates.clone(),
        vec![3.0, f64::INFINITY, 5.0, 7.0, f64::NEG_INFINITY, 9.0],
    )
    .unwrap();
    let w3 = Window::Observations { size: 3, ramp: 0 };

    for (series, window, k) in [
        (&c42, w3, None),
        (&cneg0, w3, Some(2.0)),
        (&mid_nan, w3, None),
        (&all_nan, w3, None),
        (&infs, w3, None),
        (&infs, Window::Days { size: 4, ramp: 0 }, None),
        (&x, w3, Some(0.0)),
        (&x, w3, Some(-0.0)),
        (&x, w3, Some(-2.0)),
        (&x, w3, Some(f64::NAN)),
        (&x, w3, Some(f64::INFINITY)),
        (&x, w3, Some(f64::NEG_INFINITY)),
    ] {
        let run1 = bollinger_bands(series, window, k).unwrap();
        let run2 = bollinger_bands(series, window, k).unwrap();
        assert_eq!(run1.lower.dates(), run2.lower.dates());
        assert_eq!(run1.upper.dates(), run2.upper.dates());
        assert!(run1
            .lower
            .values()
            .iter()
            .zip(run2.lower.values())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(run1
            .upper
            .values()
            .iter()
            .zip(run2.upper.values())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let neg0_out = bollinger_bands(&cneg0, w3, Some(2.0)).unwrap();
    assert!(neg0_out.lower.values()[2] == 0.0 && neg0_out.lower.values()[2].is_sign_negative());
    assert!(neg0_out.upper.values()[2] == 0.0 && neg0_out.upper.values()[2].is_sign_positive());
    let inf_obs = bollinger_bands(&infs, w3, None).unwrap();
    let inf_days = bollinger_bands(&infs, Window::Days { size: 4, ramp: 0 }, None).unwrap();
    assert!((inf_obs.lower.values()[2] - 1.1715728752538097).abs() < 1e-12);
    assert!(inf_days.lower.values().iter().all(|v| v.is_nan()));
    let k_neg = bollinger_bands(&x, w3, Some(-2.0)).unwrap();
    assert!(k_neg.lower.values()[5] > k_neg.upper.values()[5]);

    // 4. Empty -> empty across all windows; non-empty size == 0 -> Error::Window, ramp > len -> Error::Ramp
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 0, ramp: 0 },
        Window::Days { size: 2, ramp: 2 },
    ] {
        let out = bollinger_bands(&empty, w, None).unwrap();
        assert!(out.lower.values().is_empty() && out.upper.values().is_empty());
    }
    assert_eq!(
        bollinger_bands(&x, Window::Observations { size: 0, ramp: 0 }, None),
        Err(Error::Window)
    );
    assert_eq!(
        bollinger_bands(&x, Window::Days { size: 0, ramp: 0 }, None),
        Err(Error::Window)
    );
    assert_eq!(
        bollinger_bands(&x, Window::Observations { size: 2, ramp: 7 }, None),
        Err(Error::Ramp)
    );

    println!(
        "exp=({},{}) obs_k15=({},{}) leap_r3=({},{})",
        exp_def.lower.values()[5],
        exp_def.upper.values()[5],
        obs_k15.lower.values()[5],
        obs_k15.upper.values()[5],
        d3_r3.lower.values()[0],
        d3_r3.upper.values()[0]
    );
}
```

Standalone offline `cov` example for an external Cargo project:

```rust
use quantus_math::{cov, Date, Error, Series, Window};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Daily series sample covariance (ddof = 1): Expanding, Observations, and 7d Days
    let dates = vec![
        d(2019, 1, 1),
        d(2019, 1, 2),
        d(2019, 1, 3),
        d(2019, 1, 4),
        d(2019, 1, 7),
        d(2019, 1, 8),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let y = Series::try_new(dates.clone(), vec![3.5, 1.8, 2.9, 1.2, 3.1, 5.9]).unwrap();
    let exp = cov(&x, &y, Window::Expanding).unwrap();
    let obs2 = cov(&x, &y, Window::Observations { size: 2, ramp: 0 }).unwrap();
    let cal7d = cov(&x, &y, Window::Days { size: 7, ramp: 0 }).unwrap();
    assert!(exp.values()[0].is_nan());
    assert!((exp.values()[5] - 2.7).abs() < 1e-12);
    assert!((obs2.values()[5] - 4.2).abs() < 1e-12);
    assert!((cal7d.values()[5] - 3.375).abs() < 1e-12);

    // 2. Date alignment (date union for Observations vs x.dates() for Days) & leap day 2020-02-29
    let px = Series::try_new(
        vec![d(2020, 1, 1), d(2020, 1, 3), d(2020, 1, 5)],
        vec![1.0, 3.0, 5.0],
    )
    .unwrap();
    let py = Series::try_new(
        vec![
            d(2020, 1, 1),
            d(2020, 1, 2),
            d(2020, 1, 3),
            d(2020, 1, 4),
            d(2020, 1, 5),
        ],
        vec![2.0, 10.0, 6.0, 20.0, 10.0],
    )
    .unwrap();
    let align_obs = cov(&px, &py, Window::Observations { size: 5, ramp: 0 }).unwrap();
    let align_days = cov(&px, &py, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert_eq!(align_obs.dates(), py.dates());
    assert!((align_obs.values()[4] - 8.0).abs() < 1e-12);
    assert_eq!(align_days.dates(), px.dates());
    assert!((align_days.values()[2] - 4.0).abs() < 1e-12);

    let leap_dates = vec![
        d(2020, 2, 27),
        d(2020, 2, 28),
        d(2020, 2, 29),
        d(2020, 3, 2),
        d(2020, 3, 5),
    ];
    let lx = Series::try_new(leap_dates.clone(), vec![1.0, 3.0, 2.0, 6.0, 10.0]).unwrap();
    let ly = Series::try_new(leap_dates, vec![2.0, 5.0, 4.0, 11.0, 18.0]).unwrap();
    let leap_3d_r3 = cov(&lx, &ly, Window::Days { size: 3, ramp: 3 }).unwrap();
    assert_eq!(leap_3d_r3.dates(), &[d(2020, 3, 2), d(2020, 3, 5)]);
    assert!((leap_3d_r3.values()[0] - 14.0).abs() < 1e-12);
    assert!(leap_3d_r3.values()[1].is_nan());

    // 3. Constants (+0.0), large-offset stability, NaN/±Inf, overflow, and determinism
    let d5: Vec<_> = (1..=5).map(|day| d(2020, 1, day)).collect();
    let c42 = Series::try_new(d5.clone(), vec![4.2; 5]).unwrap();
    let lin5 = Series::try_new(d5.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let lo_x = Series::try_new(
        d5.clone(),
        vec![1e9 + 1.0, 1e9 + 2.0, 1e9 + 4.0, 1e9 + 7.0, 1e9 + 11.0],
    )
    .unwrap();
    let lo_y = Series::try_new(
        d5.clone(),
        vec![1e9 + 2.0, 1e9 + 5.0, 1e9 + 3.0, 1e9 + 9.0, 1e9 + 12.0],
    )
    .unwrap();
    let x_inf = Series::try_new(d5.clone(), vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0]).unwrap();
    let y_fin = Series::try_new(d5.clone(), vec![2.0, 4.0, 6.0, 8.0, 10.0]).unwrap();
    let y_nan = Series::try_new(d5.clone(), vec![2.0, f64::NAN, 6.0, 8.0, 10.0]).unwrap();
    let ov_x = Series::try_new(d5[..2].to_vec(), vec![1e308, -1e308]).unwrap();
    let ov_y = Series::try_new(d5[..2].to_vec(), vec![1e308, -1e308]).unwrap();

    for (a, b, w) in [
        (&c42, &lin5, Window::Observations { size: 3, ramp: 0 }),
        (&lo_x, &lo_y, Window::Observations { size: 3, ramp: 0 }),
        (&x_inf, &y_fin, Window::Observations { size: 3, ramp: 0 }),
        (&x_inf, &y_fin, Window::Days { size: 3, ramp: 0 }),
        (&x_inf, &y_nan, Window::Days { size: 3, ramp: 0 }),
        (&ov_x, &ov_y, Window::Observations { size: 2, ramp: 0 }),
    ] {
        let run1 = cov(a, b, w).unwrap();
        let run2 = cov(a, b, w).unwrap();
        assert_eq!(run1.dates(), run2.dates());
        assert!(run1
            .values()
            .iter()
            .zip(run2.values())
            .all(|(u, v)| u.to_bits() == v.to_bits()));
    }
    let c42_out = cov(&c42, &lin5, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!(c42_out.values()[3] == 0.0 && c42_out.values()[3].is_sign_positive());
    let lo_obs = cov(&lo_x, &lo_y, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!((lo_obs.values()[2] - (1.0 / 3.0)).abs() < 1e-12);
    let inf_obs = cov(&x_inf, &y_fin, Window::Observations { size: 3, ramp: 0 }).unwrap();
    let inf_days_fin = cov(&x_inf, &y_fin, Window::Days { size: 3, ramp: 0 }).unwrap();
    let inf_days_nan = cov(&x_inf, &y_nan, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert!((inf_obs.values()[2] - 4.0).abs() < 1e-12);
    assert!(inf_days_fin.values()[2].is_nan());
    assert!((inf_days_nan.values()[2] - 4.0).abs() < 1e-12);
    let ov_out = cov(&ov_x, &ov_y, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert!(ov_out.values()[1].is_infinite() && ov_out.values()[1].is_sign_positive());

    // 4. Empty & error boundaries
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(cov(&empty, &empty, Window::Days { size: 3, ramp: 1 })
        .unwrap()
        .values()
        .is_empty());
    assert_eq!(cov(&empty, &y, Window::Expanding), Err(Error::Window));
    assert_eq!(
        cov(&x, &y, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&x, &y, Window::Observations { size: 2, ramp: 7 }),
        Err(Error::Ramp)
    );

    println!(
        "exp={} obs2={} cal7d={} leap_r3={} lo_obs={}",
        exp.values()[5],
        obs2.values()[5],
        cal7d.values()[5],
        leap_3d_r3.values()[0],
        lo_obs.values()[4]
    );
}
```

Standalone offline `RollingLinearRegression` example (#13) for an external Cargo project:

```rust
use quantus_math::{Date, Error, RollingLinearRegression, Series};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Multi-factor rolling OLS with intercept (upstream test_rolling_linear_regression)
    let x1 = Series::try_new(
        (1..=7).map(|day| d(2019, 1, day)).collect(),
        vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, f64::NAN],
    )
    .unwrap();
    let x2 = Series::try_new(
        (1..=8).map(|day| d(2019, 1, day)).collect(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    .unwrap();
    let y = Series::try_new(
        (1..=7).map(|day| d(2019, 1, day)).collect(),
        vec![10.0, 14.0, 20.0, 28.0, 28.0, 40.0, 60.0],
    )
    .unwrap();

    let reg = RollingLinearRegression::fit(&[x1.clone(), x2.clone()], &y, 4, true).unwrap();
    assert_eq!(reg.column_indices(), &[0, 1, 2]);
    assert!(reg.coefficient(0).unwrap().values()[..3]
        .iter()
        .all(|v| v.is_nan()));
    assert!((reg.coefficient(0).unwrap().values()[5] - 19.0).abs() < 1e-10);
    assert!((reg.coefficient(1).unwrap().values()[5] - 1.0).abs() < 1e-10);
    assert!((reg.coefficient(2).unwrap().values()[5] - (-1.0)).abs() < 1e-10);
    assert!((reg.r_squared().values()[5] - 0.9019607843137254).abs() < 1e-10);
    assert!((reg.fitted_values().values()[5] - 39.0).abs() < 1e-10);
    assert!((reg.standard_deviation_of_errors().values()[5] - 4.4721359549995805).abs() < 1e-10);

    // 2. Without intercept: columns indexed 1..=k and uncentered R^2
    let no_int = RollingLinearRegression::fit(&[x1, x2], &y, 4, false).unwrap();
    assert_eq!(no_int.column_indices(), &[1, 2]);
    assert!(no_int.coefficient(0).is_none());
    assert!(no_int.coefficient(1).is_some() && no_int.coefficient(2).is_some());

    // 3. Ill-conditioned predictor (1e7 + [1..6]): Householder QR recovers true coefficients
    let d6: Vec<_> = (1..=6).map(|day| d(2020, 7, day)).collect();
    let x_ill = Series::try_new(d6.clone(), (1..=6).map(|k| 1e7 + k as f64).collect()).unwrap();
    let y_ill = Series::try_new(d6, (1..=6).map(|k| 1.0 + 2.0 * k as f64).collect()).unwrap();
    let ill = RollingLinearRegression::fit(std::slice::from_ref(&x_ill), &y_ill, 4, true).unwrap();
    assert!((ill.coefficient(0).unwrap().values()[5] - (-19_999_999.0)).abs() / 19_999_999.0 < 1e-8);
    assert!((ill.coefficient(1).unwrap().values()[5] - 2.0).abs() < 1e-8);
    assert!((ill.r_squared().values()[5] - 1.0).abs() < 1e-10);

    // 4. Error boundaries: empty predictors -> Error::Length, window <= k or aligned < window -> Error::Window
    assert_eq!(
        RollingLinearRegression::fit(&[], &y_ill, 4, true),
        Err(Error::Length)
    );
    assert_eq!(
        RollingLinearRegression::fit(std::slice::from_ref(&x_ill), &y_ill, 2, true),
        Err(Error::Window)
    );
    assert_eq!(
        RollingLinearRegression::fit(&[x_ill], &y_ill, 7, true),
        Err(Error::Window)
    );

    println!(
        "b0={} b1={} b2={} r2={} fit={} se={}",
        reg.coefficient(0).unwrap().values()[5],
        reg.coefficient(1).unwrap().values()[5],
        reg.coefficient(2).unwrap().values()[5],
        reg.r_squared().values()[5],
        reg.fitted_values().values()[5],
        reg.standard_deviation_of_errors().values()[5],
    );
}
```

Standalone offline calendar & day-count example (#14) for an external Cargo project:

```rust
use quantus_math::{
    business_day_count, business_day_offset, day_count_fraction, has_feb_29, is_business_day, Date,
    DayCountConvention, Error, HolidayCalendar, PaymentFrequency, Roll,
};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Leap-day detection and day-count fractions
    assert!(has_feb_29(d(2020, 2, 28), d(2020, 2, 29)));
    assert!(!has_feb_29(d(2020, 2, 29), d(2020, 3, 1)));
    assert!(!has_feb_29(d(1900, 1, 1), d(1900, 12, 31)));
    assert!(has_feb_29(d(2000, 1, 1), d(2000, 12, 31)));

    let dcf_ann = day_count_fraction(
        d(2020, 3, 1),
        d(2020, 9, 1),
        DayCountConvention::Actual365L,
        PaymentFrequency::Annually,
    );
    let dcf_mon = day_count_fraction(
        d(2020, 3, 1),
        d(2020, 9, 1),
        DayCountConvention::Actual365L,
        PaymentFrequency::Monthly,
    );
    assert!((dcf_ann - 184.0 / 365.0).abs() < 1e-12);
    assert!((dcf_mon - 184.0 / 366.0).abs() < 1e-12);
    assert_eq!(
        day_count_fraction(
            d(2020, 9, 1),
            d(2020, 3, 1),
            DayCountConvention::OneOne,
            PaymentFrequency::Annually,
        ),
        1.0
    );

    // 2. Explicit offline holiday calendar, rolling offsets, and asymmetric business-day counts
    let cal = HolidayCalendar::weekdays(vec![d(2020, 1, 1), d(2020, 1, 31)]);
    assert!(!is_business_day(d(2020, 1, 1), &cal));
    assert!(is_business_day(d(2020, 1, 2), &cal));

    let rolled = business_day_offset(d(2020, 1, 31), 0, Roll::ModifiedFollowing, &cal).unwrap();
    assert_eq!(rolled, d(2020, 1, 30));
    let shifted = business_day_offset(d(2020, 1, 31), 1, Roll::ModifiedFollowing, &cal).unwrap();
    assert_eq!(shifted, d(2020, 2, 3));

    assert_eq!(business_day_count(d(2020, 1, 1), d(2020, 1, 6), &cal), 2);
    assert_eq!(business_day_count(d(2020, 1, 6), d(2020, 1, 1), &cal), -3);

    // 3. Typed Error::Date boundaries
    assert_eq!(HolidayCalendar::try_new([false; 7], vec![]), Err(Error::Date));
    assert_eq!(
        business_day_offset(d(2020, 1, 1), 0, Roll::Raise, &cal),
        Err(Error::Date)
    );
    assert_eq!(
        business_day_offset(d(9999, 12, 31), 1, Roll::Forward, &cal),
        Err(Error::Date)
    );

    println!("dcf_ann={dcf_ann} dcf_mon={dcf_mon} rolled={rolled} shifted={shifted}");
}
```

Standalone offline multi-asset basket backtesting example (#15) for an external Cargo project:

```rust
use quantus_math::{backtest_basket, basket_series, BasketParam, Date, Error, RebalFreq, Series};

fn main() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();

    // 1. Two-asset basket with daily rebalancing and default equal weights / zero costs
    let dates = vec![
        d(2024, 1, 31),
        d(2024, 2, 28),
        d(2024, 2, 29),
        d(2024, 3, 29),
        d(2024, 3, 31),
        d(2024, 4, 30),
        d(2024, 5, 1),
    ];
    let asset_a = Series::try_new(
        dates.clone(),
        vec![100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 155.0],
    )
    .unwrap();
    let asset_b = Series::try_new(dates.clone(), vec![100.0; 7]).unwrap();

    let eq_nav = basket_series(
        &[asset_a.clone(), asset_b.clone()],
        None,
        None,
        RebalFreq::Daily,
    )
    .unwrap();
    assert_eq!(eq_nav.values()[0], 100.0);
    assert!((eq_nav.values()[1] - 105.0).abs() < 1e-12);

    // 2. Monthly Jan 31 leap-year anchored schedule (Feb 29, Mar 31, Apr 30) with mixed Fixed/Series params
    let w_a_series = Series::try_new(dates, vec![0.6, 0.6, 0.5, 0.5, 0.7, 0.5, 0.5]).unwrap();
    let weights = [BasketParam::Series(&w_a_series), BasketParam::Fixed(0.4)];
    let costs = [BasketParam::Fixed(0.001), BasketParam::Fixed(0.002)];
    let res = backtest_basket(
        &[asset_a.clone(), asset_b.clone()],
        Some(&weights),
        Some(&costs),
        RebalFreq::Monthly,
    )
    .unwrap();
    // Feb 28 drifts away from 0.6; Feb 29 (k=1), Mar 31 (k=2), Apr 30 (k=3) rebalance to target weights
    assert!((res.actual_weights[0].values()[1] - 0.6).abs() > 1e-3);
    assert_eq!(res.actual_weights[0].values()[2], 0.5);
    assert_eq!(res.actual_weights[0].values()[4], 0.7);
    assert_eq!(res.actual_weights[0].values()[5], 0.5);

    // 3. Typed Error::Length boundaries (empty series, length mismatch, explicit Some(&[]), disjoint dates)
    assert_eq!(backtest_basket(&[], None, None, RebalFreq::Daily), Err(Error::Length));
    assert_eq!(
        backtest_basket(&[asset_a.clone(), asset_b], Some(&[]), None, RebalFreq::Daily),
        Err(Error::Length)
    );
    let disjoint = Series::try_new(vec![d(2025, 1, 1)], vec![100.0]).unwrap();
    assert_eq!(
        backtest_basket(&[asset_a, disjoint], None, None, RebalFreq::Weekly),
        Err(Error::Length)
    );

    println!(
        "eq_final={} monthly_final={}",
        eq_nav.values().last().unwrap(),
        res.nav.values().last().unwrap()
    );
}
```

Standalone offline bucketed risk aggregation, subtraction, sorting & `point_sort_order` example (#16) for an external Cargo project:

```rust
use quantus_math::{
    aggregate_risk, point_sort_order, sort_risk, subtract_risk, Date, Error, RiskRow, RiskTable,
};

fn main() {
    let ref_date = Date::new(2024, 1, 15).unwrap();

    // 1. Market-point tenor sort order (pure std, deterministic ref_date)
    assert_eq!(point_sort_order("O/N", ref_date), Some(0.0));
    assert_eq!(point_sort_order("3x6", ref_date), Some(90.0));
    assert_eq!(point_sort_order("1y;5y", ref_date), Some(365.5));
    assert_eq!(point_sort_order("Dec68", ref_date), Some(16406.0));
    assert_eq!(point_sort_order("USD Rho", ref_date), None);

    let row = |pt: &str, v: f64| RiskRow {
        date: None,
        time: None,
        mkt_type: "IR".into(),
        mkt_asset: "USD".into(),
        mkt_class: "SWAP".into(),
        mkt_point: pt.into(),
        value: v,
    };

    // 2. Stable sort_risk preserves input order on tied points ("30d" and "1m" both 30.0)
    let t1 = RiskTable {
        unit: Some("USD".into()),
        rows: vec![row("5y", 10.0), row("30d", 5.0), row("1m", 7.0), row("1y", 20.0)],
    };
    let sorted = sort_risk(&t1, ref_date);
    assert_eq!(
        sorted.rows.iter().map(|r| r.mkt_point.as_str()).collect::<Vec<_>>(),
        vec!["30d", "1m", "1y", "5y"]
    );

    // 3. aggregate_risk: fillna(0), strict abs(value) > threshold, unit inheritance, lex tie-break
    let t2 = RiskTable {
        unit: None,
        rows: vec![row("5y", -10.0), row("1y", f64::NAN), row("2y", -15.0)],
    };
    let agg = aggregate_risk(&[t1.clone(), t2.clone()], Some(0.0), ref_date).unwrap();
    assert_eq!(agg.unit.as_deref(), Some("USD"));
    assert_eq!(
        agg.rows.iter().map(|r| (r.mkt_point.as_str(), r.value)).collect::<Vec<_>>(),
        vec![("1m", 7.0), ("30d", 5.0), ("1y", 20.0), ("2y", -15.0)]
    );

    // 4. subtract_risk and typed Error::UnitMismatch / Error::Length boundaries
    let diff = subtract_risk(&t1, &t2, ref_date).unwrap();
    assert_eq!(diff.rows.last().unwrap().value, 20.0); // 10.0 - (-10.0)

    let eur = RiskTable { unit: Some("EUR".into()), rows: vec![row("5y", 1.0)] };
    assert_eq!(aggregate_risk(&[t1.clone(), eur.clone()], None, ref_date), Err(Error::UnitMismatch));
    assert_eq!(subtract_risk(&t1, &eur, ref_date), Err(Error::UnitMismatch));
    assert_eq!(aggregate_risk(&[], None, ref_date), Err(Error::Length));

    println!("agg_rows={} diff_5y={}", agg.rows.len(), diff.rows.last().unwrap().value);
}
```

Standalone offline `vol_swap_volatility` & `corr_swap_correlation` example (#17) for an external Cargo project:

```rust
use quantus_math::{corr_swap_correlation, vol_swap_volatility, Date, Error, Series};

fn main() {
    let d = |m, day| Date::new(2020, m, day).unwrap();
    // Daily close prices crossing leap day 2020-02-29
    let dates = vec![d(2, 27), d(2, 28), d(2, 29), d(3, 1), d(3, 2), d(3, 3)];
    let x = Series::try_new(
        dates.clone(),
        vec![100.0, 102.0, 101.0, 104.0, 103.0, 106.0],
    )
    .unwrap();
    let y = Series::try_new(dates, vec![50.0, 51.5, 50.2, 52.0, 52.5, 54.0]).unwrap();

    // 1. Volatility swap volatility (% annualized): zero-mean RMS vs sample std, 252 vs 365
    let vol_zm = vol_swap_volatility(&x, Some(4), None, true).unwrap();
    let vol_365 = vol_swap_volatility(&x, Some(4), Some(365), true).unwrap();
    let vol_sample = vol_swap_volatility(&x, Some(4), None, false).unwrap();
    assert_eq!(vol_zm.dates(), &[d(3, 1), d(3, 2), d(3, 3)]);
    assert!(((vol_365.values()[0] / vol_zm.values()[0]) - (365.0_f64 / 252.0_f64).sqrt()).abs() < 1e-12);
    assert!((vol_zm.values()[0] - vol_sample.values()[0]).abs() > 1e-3);

    // 2. Correlation swap correlation (dimensionless ratio in [-1, 1]): zero-mean vs Pearson
    let corr_zm = corr_swap_correlation(&x, &y, Some(4), true).unwrap();
    let corr_pearson = corr_swap_correlation(&x, &y, Some(4), false).unwrap();
    assert_eq!(corr_zm.dates(), &[d(2, 29), d(3, 1), d(3, 2), d(3, 3)]);
    assert!(corr_zm.values()[0].is_nan()); // masked by inner ramp on aligned log returns
    assert!(corr_zm.values()[1..].iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
    assert!(corr_pearson.values().iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));

    // 3. Default full-series n_days = None and typed error boundaries
    let full_vol = vol_swap_volatility(&x, None, None, true).unwrap();
    assert_eq!(full_vol.dates(), &[d(3, 3)]);
    assert_eq!(vol_swap_volatility(&x, Some(0), None, true), Err(Error::Window));
    assert_eq!(corr_swap_correlation(&x, &y, Some(1), true), Err(Error::Window));

    println!(
        "vol_zm={} vol_sample={} corr_zm={} corr_pearson={}",
        vol_zm.values().last().unwrap(),
        vol_sample.values().last().unwrap(),
        corr_zm.values().last().unwrap(),
        corr_pearson.values().last().unwrap(),
    );
}
```

Standalone offline event-driven `PredefinedAssetEngine` example (#18) for an external Cargo project:

```rust
use std::collections::BTreeMap;
use quantus_math::{
    run_predefined_backtest, Date, Error, HolidayCalendar, PredefinedBacktestInput,
    PredefinedOrder, PredefinedOrderKind, Series, Timestamp, TradeStatus,
};

fn main() {
    let d = |day| Date::new(2021, 1, day).unwrap();
    let ts = |day, h, m, s| Timestamp::new(d(day), h, m, s).unwrap();
    let cal = HolidayCalendar::weekdays(vec![]);

    let mut daily_prices = BTreeMap::new();
    daily_prices.insert(
        "TestRic".to_string(),
        Series::try_new(vec![d(4), d(5), d(6)], vec![1.0, 1.5, 2.0]).unwrap(),
    );

    // 1. Flagship MOC backtest (upstream test_backtest_predefined)
    let moc = run_predefined_backtest(&PredefinedBacktestInput {
        start: d(4),
        end: d(6),
        calendar: &cal,
        initial_value: 100.0,
        eod_valuation_time: (23, 0, 0),
        cash_label: "USD".to_string(),
        daily_prices: daily_prices.clone(),
        intraday_prices: BTreeMap::new(),
        orders: vec![PredefinedOrder {
            instrument: "TestRic".to_string(),
            quantity: 1.0,
            generation_time: ts(5, 10, 0, 0),
            kind: PredefinedOrderKind::MarketOnClose {
                execution_date: d(5),
            },
            source: "Test".to_string(),
        }],
    })
    .unwrap();
    assert_eq!(moc.performance.values(), &[100.0, 100.0, 100.5]);
    assert_eq!(moc.snapshots[0].holdings["USD"], 98.5);
    assert_eq!(moc.snapshots[0].holdings["TestRic"], 1.0);
    assert_eq!(moc.trade_ledger[0].status, TradeStatus::Open);

    // 2. Intraday TWAP half-open (start, end] round-trip + Cost fee
    let mut intraday_prices = BTreeMap::new();
    intraday_prices.insert(
        "TestRic".to_string(),
        vec![
            (ts(5, 10, 0, 0), 999.0), // exact start: excluded
            (ts(5, 10, 15, 0), 15.0),
            (ts(5, 10, 30, 0), 17.0), // exact end: included -> mean 16.0
            (ts(5, 14, 30, 0), 25.0),
        ],
    );
    let twap = run_predefined_backtest(&PredefinedBacktestInput {
        start: d(4),
        end: d(5),
        calendar: &cal,
        initial_value: 100.0,
        eod_valuation_time: (23, 0, 0),
        cash_label: "USD".to_string(),
        daily_prices,
        intraday_prices,
        orders: vec![
            PredefinedOrder {
                instrument: "TestRic".to_string(),
                quantity: 1.0,
                generation_time: ts(5, 9, 30, 0),
                kind: PredefinedOrderKind::Twap {
                    start: ts(5, 10, 0, 0),
                    end: ts(5, 10, 30, 0),
                },
                source: "TwapEntry".to_string(),
            },
            PredefinedOrder {
                instrument: "TestRic".to_string(),
                quantity: -1.0,
                generation_time: ts(5, 9, 30, 0),
                kind: PredefinedOrderKind::Twap {
                    start: ts(5, 14, 0, 0),
                    end: ts(5, 14, 30, 0),
                },
                source: "TwapExit".to_string(),
            },
            PredefinedOrder {
                instrument: "USD".to_string(),
                quantity: -1.0,
                generation_time: ts(5, 15, 0, 0),
                kind: PredefinedOrderKind::Cost {
                    execution_time: ts(5, 15, 0, 0),
                },
                source: "Fee".to_string(),
            },
        ],
    })
    .unwrap();
    assert_eq!(twap.performance.values(), &[100.0, 108.0]);
    assert!(!twap.snapshots[0].holdings.contains_key("TestRic"));
    assert_eq!(twap.snapshots[0].weights["USD"], 1.0);
    assert_eq!(twap.trade_ledger[0].status, TradeStatus::Closed);
    assert_eq!(twap.trade_ledger[0].trade_pnl, Some(9.0));

    // 3. Typed validation error
    assert_eq!(Timestamp::new(d(5), 24, 0, 0), Err(Error::Date));

    println!(
        "moc_final={} twap_final={}",
        moc.performance.values().last().unwrap(),
        twap.performance.values().last().unwrap()
    );
}
```

Parity: `python3 parity/check.py --replay`, `python3 parity/check-volatility.py --replay`, `python3 parity/check-correlation.py --replay`, `python3 parity/check-beta.py --replay`, `python3 parity/check-drawdown.py --replay`, `python3 parity/check-ema.py --replay`, `python3 parity/check-smma.py --replay`, `python3 parity/check-rsi.py --replay`, `python3 parity/check-macd.py --replay`, `python3 parity/check-prices.py --replay`, `python3 parity/check-std.py --replay`, `python3 parity/check-bollinger-bands.py --replay`, `python3 parity/check-cov.py --replay`, `python3 parity/check-rolling-ols.py --replay`, `python3 parity/check-calendar.py --replay`, `python3 parity/check-basket.py --replay`, `python3 parity/check-risk.py --replay`, `python3 parity/check-volcorr-swap.py --replay`, and `python3 parity/check-predefined.py --replay` build the offline replay binary and check frozen unrounded Python outputs. [The assertion map](parity/MAP.md) records the covered cases and exclusions. To regenerate vectors, use the pinned Python environment and an independently verified checkout of the upstream SHA:

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
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-std.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-bollinger-bands.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-cov.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-rolling-ols.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-calendar.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-basket.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-risk.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-volcorr-swap.py /path/to/gs-quant
PYTHONDONTWRITEBYTECODE=1 /path/to/python parity/generate-predefined.py /path/to/gs-quant
```

Generation imports the upstream Python implementation; replay does not. Compare a regenerated file and its hash before accepting any change. Upstream LICENSE, NOTICE and NOTICE.txt are preserved; source attribution is in `src/lib.rs`.
