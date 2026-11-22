//! Offline date-indexed prices, price level reconstruction from returns, moving averages, exponential moving averages, smoothed moving averages, relative strength index, moving average convergence divergence (MACD), rolling standard deviation (`std`), rolling sample covariance (`cov`), rolling linear regression (`RollingLinearRegression`), Bollinger bands (`bollinger_bands`), returns, realized volatility, correlation, beta, maximum drawdown, business-day calendar / day-count utilities (`HolidayCalendar`, `DayCountConvention`, `PaymentFrequency`, `Roll`, `has_feb_29`, `day_count_fraction`, `is_business_day`, `business_day_offset`, `business_day_count`), offline multi-asset basket backtesting (`RebalFreq`, `BasketParam`, `BasketResult`, `backtest_basket`, `basket_series`), and offline bucketed risk aggregation, subtraction, sorting, and market-point sort order (`RiskRow`, `RiskTable`, `point_sort_order`, `sort_risk`, `aggregate_risk`, `subtract_risk`).
//! Adapted from gs_quant/timeseries/{technicals,econometrics,statistics,datetime,backtesting,helper}.py,
//! gs_quant/datetime/{date,point}.py (with explicit-holiday `np.busdaycalendar` semantics), and
//! gs_quant/risk/core.py at 70220e300524d54bd273be0f4c485d53b3780482; Rust translation and
//! modifications by STOA.
//! Copyright 2018-2019 Goldman Sachs. Upstream also includes code copyright Scott Weinstein.
//! Licensed under Apache-2.0; see LICENSE and both NOTICE files.

use std::collections::BTreeMap;
use std::fmt;

const MIN_ORDINAL: i64 = 1; // 0001-01-01 (Monday)
const MAX_ORDINAL: i64 = 3_652_059; // 9999-12-31 (Friday)

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// A Gregorian calendar date without a time or time zone.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Date {
    year: i32,
    month: u8,
    day: u8,
}
impl Date {
    /// Validate a date in years 1 through 9999.
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self, Error> {
        let leap = is_leap_year(year);
        let max = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if leap => 29,
            2 => 28,
            _ => 0,
        };
        if !(1..=9999).contains(&year) || day == 0 || day > max {
            return Err(Error::Date);
        }
        Ok(Self { year, month, day })
    }
    /// Calendar year (`1..=9999`).
    pub fn year(&self) -> i32 {
        self.year
    }
    /// Calendar month (`1..=12`).
    pub fn month(&self) -> u8 {
        self.month
    }
    /// Day of month (`1..=31`).
    pub fn day(&self) -> u8 {
        self.day
    }
    fn ordinal(self) -> i64 {
        let y = i64::from(self.year - 1);
        let mut n = 365 * y + y / 4 - y / 100 + y / 400 + i64::from(self.day);
        for m in 1..self.month {
            n += match m {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                _ if is_leap_year(self.year) => 29,
                _ => 28,
            };
        }
        n
    }
    /// Inverse of `ordinal()` in O(1) using Howard Hinnant's proleptic Gregorian civil-from-days
    /// algorithm, bounded to `0001-01-01..=9999-12-31`.
    fn from_ordinal(ordinal: i64) -> Result<Self, Error> {
        if !(MIN_ORDINAL..=MAX_ORDINAL).contains(&ordinal) {
            return Err(Error::Date);
        }
        // Days since proleptic 0000-03-01 (0001-01-01 is day 306, ordinal 1).
        let z = ordinal + 305;
        let era = z / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u8;
        let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u8;
        let year = (if month <= 2 { y + 1 } else { y }) as i32;
        Ok(Self { year, month, day })
    }
    /// ISO weekday index (`Monday = 0` .. `Sunday = 6`), matching Python `date.weekday()`.
    fn weekday(self) -> usize {
        ((self.ordinal() - 1).rem_euclid(7)) as usize
    }
}
impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Invalid date, series or calculation argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Date,
    Length,
    UnsortedOrDuplicate,
    Window,
    Ramp,
    Lag,
    NonPositivePrice,
    AnnualizationFactor,
    Beta,
    Span,
    Overflow,
    UnitMismatch,
}

/// Strictly increasing calendar dates and floating point values. NaN represents missing data.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    dates: Vec<Date>,
    values: Vec<f64>,
}
impl Series {
    /// Dates in strictly increasing order.
    pub fn dates(&self) -> &[Date] {
        &self.dates
    }
    /// Values in the same order as dates; NaN represents missing data.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Reject different lengths and non-increasing or duplicate dates.
    pub fn try_new(dates: Vec<Date>, values: Vec<f64>) -> Result<Self, Error> {
        if dates.len() != values.len() {
            return Err(Error::Length);
        }
        if dates.windows(2).any(|x| x[0] >= x[1]) {
            return Err(Error::UnsortedOrDuplicate);
        }
        Ok(Self { dates, values })
    }
}
/// Window size and ramp measured in observations or calendar days respectively.
#[derive(Debug, Clone, Copy)]
pub enum Window {
    Expanding,
    Observations { size: usize, ramp: usize },
    Days { size: u32, ramp: u32 },
}
/// Return units: simple and logarithmic are dimensionless; absolute uses price units.
#[derive(Debug, Clone, Copy)]
pub enum ReturnKind {
    Simple,
    Logarithmic,
    Absolute,
}

/// Mean of available observations. NaNs are ignored; an all-NaN window yields NaN.
/// Empty input returns empty before window validation, matching the upstream function.
pub fn moving_average(x: &Series, window: Window) -> Result<Series, Error> {
    if x.dates.is_empty() {
        return Series::try_new(vec![], vec![]);
    }
    match window {
        Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
            return Err(Error::Window)
        }
        Window::Observations { ramp, .. } if ramp > x.dates.len() => return Err(Error::Ramp),
        Window::Observations { size, .. } if size > x.dates.len() => {
            return Series::try_new(vec![], vec![])
        }
        _ => {}
    }
    let mut dates = Vec::new();
    let mut values = Vec::new();
    for i in 0..x.dates.len() {
        let include = match window {
            Window::Expanding => true,
            Window::Observations { ramp, .. } => i >= ramp,
            Window::Days { ramp, .. } => {
                x.dates[i].ordinal() - x.dates[0].ordinal() >= i64::from(ramp)
            }
        };
        if !include {
            continue;
        }
        let start = match window {
            Window::Expanding => 0,
            Window::Observations { size, .. } => (i + 1).saturating_sub(size),
            Window::Days { size, .. } => (0..=i)
                .find(|&j| x.dates[i].ordinal() - x.dates[j].ordinal() < i64::from(size))
                .unwrap_or(i),
        };
        let mut sum = 0.0;
        let mut count = 0;
        for &v in &x.values[start..=i] {
            if !v.is_nan() {
                sum += v;
                count += 1;
            }
        }
        dates.push(x.dates[i]);
        values.push(if count == 0 {
            f64::NAN
        } else {
            sum / f64::from(count)
        });
    }
    Series::try_new(dates, values)
}
/// Observation-lag returns. Lag must be positive. NaN propagates; logarithms require positive non-NaN prices.
pub fn returns(x: &Series, lag: usize, kind: ReturnKind) -> Result<Series, Error> {
    if lag == 0 {
        return Err(Error::Lag);
    }
    if matches!(kind, ReturnKind::Logarithmic) && x.values.iter().any(|v| !v.is_nan() && *v <= 0.0)
    {
        return Err(Error::NonPositivePrice);
    }
    let values = x
        .values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i < lag {
                f64::NAN
            } else {
                let prev = x.values[i - lag];
                match kind {
                    ReturnKind::Simple => v / prev - 1.0,
                    ReturnKind::Logarithmic => v.ln() - prev.ln(),
                    ReturnKind::Absolute => v - prev,
                }
            }
        })
        .collect();
    Series::try_new(x.dates.clone(), values)
}

/// Reconstruct price levels from a returns series.
///
/// Mirrors upstream `gs_quant/timeseries/econometrics.py:prices` (lines 353–421) and
/// `gs_quant/timeseries/statistics.py:{product,sum_}`:
///
/// - Defaults `initial = 1.0` when `None` and `kind = ReturnKind::Simple` when `None`.
/// - Empty input returns an empty `Series` immediately, regardless of `initial` or `kind`.
/// - [`ReturnKind::Simple`]: multiplies finite factors `1.0 + v` cumulatively starting from
///   `1.0` (skipping `NaN` and `±Inf`), and emits `prod * initial` at each date.
/// - [`ReturnKind::Logarithmic`]: rejects any finite element `v` where `v.exp().is_infinite()`
///   with `Err(Error::Overflow)` (matching Python `math.exp` `OverflowError`). `+Inf` is skipped
///   without error (`exp(+Inf)` is non-finite), `-Inf` and large negative values underflow to
///   `0.0` (collapsing the product to `0.0`), and cumulative product overflow of finite
///   exponentials (e.g. `[709.0, 709.0]`) produces `+Inf` without error. Emits `prod * initial`.
/// - [`ReturnKind::Absolute`]: sums finite elements (`v.is_finite()`, skipping `NaN` and
///   `±Inf`). Before any finite observation, emits `+0.0 + initial`; once a finite observation is
///   seen, accumulates from `-0.0` (preserving `-0.0` when all finite values are `-0.0`) and emits
///   `sum + initial`.
pub fn prices(x: &Series, initial: Option<f64>, kind: Option<ReturnKind>) -> Result<Series, Error> {
    if x.dates.is_empty() {
        return Series::try_new(vec![], vec![]);
    }
    let initial = initial.unwrap_or(1.0);
    let kind = kind.unwrap_or(ReturnKind::Simple);
    let mut values = Vec::with_capacity(x.values.len());
    match kind {
        ReturnKind::Simple => {
            let mut prod = 1.0_f64;
            for &v in &x.values {
                let factor = 1.0 + v;
                if factor.is_finite() {
                    prod *= factor;
                }
                values.push(prod * initial);
            }
        }
        ReturnKind::Logarithmic => {
            for &v in &x.values {
                if v.is_finite() && v.exp().is_infinite() {
                    return Err(Error::Overflow);
                }
            }
            let mut prod = 1.0_f64;
            for &v in &x.values {
                let factor = v.exp();
                if factor.is_finite() {
                    prod *= factor;
                }
                values.push(prod * initial);
            }
        }
        ReturnKind::Absolute => {
            let mut has_obs = false;
            let mut sum = -0.0_f64;
            for &v in &x.values {
                if v.is_finite() {
                    sum += v;
                    has_obs = true;
                }
                let acc = if has_obs { sum } else { 0.0 };
                values.push(acc + initial);
            }
        }
    }
    Series::try_new(x.dates.clone(), values)
}

fn auto_annualization_factor(dates: &[Date]) -> Result<f64, Error> {
    if dates.len() < 2 {
        return Err(Error::AnnualizationFactor);
    }
    let avg_days =
        (dates[dates.len() - 1].ordinal() - dates[0].ordinal()) as f64 / (dates.len() - 1) as f64;
    if avg_days < 2.1 {
        Ok(252.0)
    } else if (6.0..8.0).contains(&avg_days) {
        Ok(52.0)
    } else if (14.0..17.0).contains(&avg_days) {
        Ok(26.0)
    } else if (25.0..35.0).contains(&avg_days) {
        Ok(12.0)
    } else if (85.0..97.0).contains(&avg_days) {
        Ok(4.0)
    } else if (360.0..386.0).contains(&avg_days) {
        Ok(1.0)
    } else {
        Err(Error::AnnualizationFactor)
    }
}

/// Annualized realized volatility in percent over a rolling or expanding window.
///
/// When `returns_kind` is `Some(kind)`, computes 1-observation returns from `x`; when `None`,
/// treats `x` as a returns series. When `assume_zero_mean` is `false`, uses unbiased sample
/// standard deviation (`n - 1` denominator); when `true`, uses zero-mean RMS (`n` denominator).
/// When `annualization_factor` is `None`, classifies the factor from average calendar-day spacing
/// (`252`, `52`, `26`, `12`, `4`, or `1`). Empty input returns empty before window validation.
pub fn volatility(
    x: &Series,
    window: Window,
    returns_kind: Option<ReturnKind>,
    annualization_factor: Option<u32>,
    assume_zero_mean: bool,
) -> Result<Series, Error> {
    if x.dates.is_empty() {
        return Series::try_new(vec![], vec![]);
    }
    match window {
        Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
            return Err(Error::Window)
        }
        Window::Observations { ramp, .. } if ramp > x.dates.len() => return Err(Error::Ramp),
        _ => {}
    }
    let computed;
    let ret = match returns_kind {
        Some(kind) => {
            computed = returns(x, 1, kind)?;
            &computed
        }
        None => x,
    };
    if matches!(window, Window::Observations { size, .. } if size > ret.dates.len()) {
        return if annualization_factor.is_some() {
            Series::try_new(vec![], vec![])
        } else {
            Err(Error::AnnualizationFactor)
        };
    }
    let factor = match annualization_factor {
        Some(factor) => f64::from(factor),
        None => auto_annualization_factor(&ret.dates)?,
    };
    let factor_sqrt = factor.sqrt();
    let mut dates = Vec::new();
    let mut values = Vec::new();
    for i in 0..ret.dates.len() {
        let include = match window {
            Window::Expanding => true,
            Window::Observations { ramp, .. } => i >= ramp,
            Window::Days { ramp, .. } => {
                ret.dates[i].ordinal() - ret.dates[0].ordinal() >= i64::from(ramp)
            }
        };
        if !include {
            continue;
        }
        let start = match window {
            Window::Expanding => 0,
            Window::Observations { size, .. } => (i + 1).saturating_sub(size),
            Window::Days { size, .. } => (0..=i)
                .find(|&j| ret.dates[i].ordinal() - ret.dates[j].ordinal() < i64::from(size))
                .unwrap_or(i),
        };
        let vol = if assume_zero_mean {
            let mut sum_sq = 0.0;
            let mut count = 0usize;
            for &v in &ret.values[start..=i] {
                if !v.is_nan() {
                    sum_sq += v * v;
                    count += 1;
                }
            }
            if count == 0 {
                f64::NAN
            } else {
                (sum_sq / count as f64).sqrt()
            }
        } else {
            let mut sum = 0.0;
            let mut count = 0usize;
            for &v in &ret.values[start..=i] {
                if !v.is_nan() {
                    sum += v;
                    count += 1;
                }
            }
            if count < 2 {
                f64::NAN
            } else {
                let mean = sum / count as f64;
                let mut sum_sq = 0.0;
                for &v in &ret.values[start..=i] {
                    if !v.is_nan() {
                        let d = v - mean;
                        sum_sq += d * d;
                    }
                }
                (sum_sq / (count - 1) as f64).sqrt()
            }
        };
        dates.push(ret.dates[i]);
        values.push(vol * factor_sqrt * 100.0);
    }
    Series::try_new(dates, values)
}

/// Rolling or expanding correlation between two date-indexed series.
///
/// Returns a dimensionless ratio (not percent) indexed by `x`'s dates after applying the outer
/// ramp. When `returns_kind` is `Some((kind_x, kind_y))`, computes lag-1 returns on each series
/// before date intersection; when `None`, treats both inputs as precomputed returns. Common dates
/// where both values are non-NaN form the aligned pairs used for windowing.
///
/// When `assume_zero_mean` is `false`, computes sample Pearson correlation (`NaN` for `< 2`
/// pairs or zero variance). When `assume_zero_mean` is `true`, computes zero-mean normalized
/// inner-product correlation via upstream's `mean` and zero-mean `volatility` pipeline on the
/// aligned pairs—preserving automatic annualization errors, oversize-window errors, and inner ramp
/// masking before reindexing to `x` and applying the outer ramp. Empty `x` returns empty before
/// window validation.
pub fn correlation(
    x: &Series,
    y: &Series,
    window: Window,
    returns_kind: Option<(ReturnKind, ReturnKind)>,
    assume_zero_mean: bool,
) -> Result<Series, Error> {
    if x.dates.is_empty() {
        return Series::try_new(vec![], vec![]);
    }
    match window {
        Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
            return Err(Error::Window)
        }
        Window::Observations { ramp, .. } if ramp > x.dates.len() => return Err(Error::Ramp),
        _ => {}
    }
    let computed_x;
    let computed_y;
    let (ret_1, ret_2) = match returns_kind {
        Some((kind_x, kind_y)) => {
            computed_x = returns(x, 1, kind_x)?;
            computed_y = returns(y, 1, kind_y)?;
            (&computed_x, &computed_y)
        }
        None => (x, y),
    };

    let mut clean_dates = Vec::new();
    let mut clean_r1 = Vec::new();
    let mut clean_r2 = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < ret_1.dates.len() && j < ret_2.dates.len() {
        match ret_1.dates[i].cmp(&ret_2.dates[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let v1 = ret_1.values[i];
                let v2 = ret_2.values[j];
                if !v1.is_nan() && !v2.is_nan() {
                    clean_dates.push(ret_1.dates[i]);
                    clean_r1.push(v1);
                    clean_r2.push(v2);
                }
                i += 1;
                j += 1;
            }
        }
    }

    let (corr_dates, corr_values) = if assume_zero_mean {
        let inner_w = match window {
            Window::Expanding => Window::Observations {
                size: x.dates.len(),
                ramp: 0,
            },
            w => w,
        };
        let prod_values = clean_r1
            .iter()
            .zip(&clean_r2)
            .map(|(&a, &b)| a * b)
            .collect();
        let prod = Series::try_new(clean_dates.clone(), prod_values)?;
        let clean_1 = Series::try_new(clean_dates.clone(), clean_r1)?;
        let clean_2 = Series::try_new(clean_dates, clean_r2)?;

        let zero_mean_cov = moving_average(&prod, inner_w)?;
        let vols_1 = volatility(&clean_1, inner_w, None, None, true)?;
        let factor = auto_annualization_factor(&clean_1.dates)?;
        let vols_2 = volatility(&clean_2, inner_w, None, None, true)?;
        let scale = factor.sqrt() * 100.0;
        let values = zero_mean_cov
            .values
            .iter()
            .zip(&vols_1.values)
            .zip(&vols_2.values)
            .map(|((&cov, &v1), &v2)| cov / ((v1 / scale) * (v2 / scale)))
            .collect();
        (zero_mean_cov.dates, values)
    } else {
        let mut values = Vec::with_capacity(clean_dates.len());
        for idx in 0..clean_dates.len() {
            let start = match window {
                Window::Expanding => 0,
                Window::Observations { size, .. } => (idx + 1).saturating_sub(size),
                Window::Days { size, .. } => (0..=idx)
                    .find(|&k| {
                        clean_dates[idx].ordinal() - clean_dates[k].ordinal() < i64::from(size)
                    })
                    .unwrap_or(idx),
            };
            let n = idx + 1 - start;
            let wx = &clean_r1[start..=idx];
            let wy = &clean_r2[start..=idx];
            let is_const = |w: &[f64]| w[0].is_finite() && w[1..].iter().all(|&v| v == w[0]);
            if n < 2 || is_const(wx) || is_const(wy) {
                values.push(f64::NAN);
            } else {
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                for (&vx, &vy) in wx.iter().zip(wy) {
                    sum_x += vx;
                    sum_y += vy;
                }
                let mean_x = sum_x / n as f64;
                let mean_y = sum_y / n as f64;
                let mut sxx = 0.0;
                let mut syy = 0.0;
                let mut sxy = 0.0;
                for (&vx, &vy) in wx.iter().zip(wy) {
                    let dx = vx - mean_x;
                    let dy = vy - mean_y;
                    sxx += dx * dx;
                    syy += dy * dy;
                    sxy += dx * dy;
                }
                if sxx == 0.0 || syy == 0.0 {
                    values.push(f64::NAN);
                } else {
                    values.push(sxy / (sxx.sqrt() * syy.sqrt()));
                }
            }
        }
        (clean_dates, values)
    };

    if matches!(window, Window::Observations { size, .. } if size > x.dates.len()) {
        return Series::try_new(vec![], vec![]);
    }
    let mut out_dates = Vec::new();
    let mut out_values = Vec::new();
    let mut k = 0;
    for (idx, &date) in x.dates.iter().enumerate() {
        while k < corr_dates.len() && corr_dates[k] < date {
            k += 1;
        }
        let value = if k < corr_dates.len() && corr_dates[k] == date {
            corr_values[k]
        } else {
            f64::NAN
        };
        let include = match window {
            Window::Expanding => true,
            Window::Observations { ramp, .. } => idx >= ramp,
            Window::Days { ramp, .. } => date.ordinal() - x.dates[0].ordinal() >= i64::from(ramp),
        };
        if include {
            out_dates.push(date);
            out_values.push(value);
        }
    }
    Series::try_new(out_dates, out_values)
}

/// Rolling or expanding beta of `x` against `benchmark`.
///
/// When `prices` is `true`, computes lag-1 simple returns on `x` and `benchmark` independently
/// before windowing; when `false`, treats both inputs as precomputed returns.
///
/// For `Window::Expanding` and `Window::Observations`, the numerator is rolling sample covariance
/// (`n - 1` denominator, skipping NaNs) over the outer-joined date union (using window size
/// `x.dates().len()` for `Window::Expanding`), and the denominator is rolling sample variance
/// (`n - 1` denominator, skipping NaNs) computed on `benchmark`'s own dates, divided by date on
/// the union index. For `Window::Days`, both covariance and variance are computed over
/// `(date - size, date]` on the date intersection of `x` and `benchmark`, propagating any `NaN`
/// inside a window.
///
/// An exactly constant finite benchmark window yields `NaN` (zero denominator). The first three
/// positions of the intermediate result are masked to `NaN` before reindexing onto `x`'s dates
/// (`NaN` for unmatched dates) and applying the outer ramp. Empty `x` returns empty before window
/// validation.
pub fn beta(x: &Series, benchmark: &Series, window: Window, prices: bool) -> Result<Series, Error> {
    if x.dates.is_empty() {
        return Series::try_new(vec![], vec![]);
    }
    match window {
        Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
            return Err(Error::Window)
        }
        Window::Observations { ramp, .. } if ramp > x.dates.len() => return Err(Error::Ramp),
        Window::Observations { size, .. } if size > x.dates.len() => {
            return Series::try_new(vec![], vec![])
        }
        _ => {}
    }

    let computed_x;
    let computed_b;
    let (ret_x, ret_b) = if prices {
        computed_x = returns(x, 1, ReturnKind::Simple)?;
        computed_b = returns(benchmark, 1, ReturnKind::Simple)?;
        (&computed_x, &computed_b)
    } else {
        (x, benchmark)
    };

    let (res_dates, mut res_values) = match window {
        Window::Days { size, .. } => {
            let mut inter_dates = Vec::new();
            let mut inter_x = Vec::new();
            let mut inter_b = Vec::new();
            let mut i = 0;
            let mut j = 0;
            while i < ret_x.dates.len() && j < ret_b.dates.len() {
                match ret_x.dates[i].cmp(&ret_b.dates[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        inter_dates.push(ret_x.dates[i]);
                        inter_x.push(ret_x.values[i]);
                        inter_b.push(ret_b.values[j]);
                        i += 1;
                        j += 1;
                    }
                }
            }

            let mut values = Vec::with_capacity(inter_dates.len());
            let mut start = 0;
            for idx in 0..inter_dates.len() {
                while start < idx
                    && inter_dates[idx].ordinal() - inter_dates[start].ordinal() >= i64::from(size)
                {
                    start += 1;
                }
                let wx = &inter_x[start..=idx];
                let wb = &inter_b[start..=idx];
                let n = idx + 1 - start;
                let has_nan = wx.iter().any(|v| v.is_nan()) || wb.iter().any(|v| v.is_nan());
                let const_b = wb[0].is_finite() && wb[1..].iter().all(|&v| v == wb[0]);
                if n < 2 || has_nan || const_b {
                    values.push(f64::NAN);
                } else {
                    let mean_x = wx.iter().sum::<f64>() / n as f64;
                    let mean_b = wb.iter().sum::<f64>() / n as f64;
                    let mut sxb = 0.0;
                    let mut sbb = 0.0;
                    for (&vx, &vb) in wx.iter().zip(wb) {
                        let dx = vx - mean_x;
                        let db = vb - mean_b;
                        sxb += dx * db;
                        sbb += db * db;
                    }
                    values.push(if sbb == 0.0 { f64::NAN } else { sxb / sbb });
                }
            }
            (inter_dates, values)
        }
        Window::Expanding | Window::Observations { .. } => {
            let win_size = match window {
                Window::Expanding => x.dates.len(),
                Window::Observations { size, .. } => size,
                Window::Days { .. } => unreachable!(),
            };

            let mut union_dates = Vec::with_capacity(ret_x.dates.len() + ret_b.dates.len());
            let mut union_x = Vec::with_capacity(ret_x.dates.len() + ret_b.dates.len());
            let mut union_b = Vec::with_capacity(ret_x.dates.len() + ret_b.dates.len());
            let mut i = 0;
            let mut j = 0;
            while i < ret_x.dates.len() || j < ret_b.dates.len() {
                let ord = if i < ret_x.dates.len() && j < ret_b.dates.len() {
                    ret_x.dates[i].cmp(&ret_b.dates[j])
                } else if i < ret_x.dates.len() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
                match ord {
                    std::cmp::Ordering::Less => {
                        union_dates.push(ret_x.dates[i]);
                        union_x.push(ret_x.values[i]);
                        union_b.push(f64::NAN);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        union_dates.push(ret_b.dates[j]);
                        union_x.push(f64::NAN);
                        union_b.push(ret_b.values[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        union_dates.push(ret_x.dates[i]);
                        union_x.push(ret_x.values[i]);
                        union_b.push(ret_b.values[j]);
                        i += 1;
                        j += 1;
                    }
                }
            }

            let mut var_b = Vec::with_capacity(ret_b.dates.len());
            for j in 0..ret_b.dates.len() {
                let start = (j + 1).saturating_sub(win_size);
                let mut count = 0usize;
                let mut sum = 0.0;
                let mut first = f64::NAN;
                let mut all_equal = true;
                for &v in &ret_b.values[start..=j] {
                    if !v.is_nan() {
                        if count == 0 {
                            first = v;
                        } else if v != first {
                            all_equal = false;
                        }
                        sum += v;
                        count += 1;
                    }
                }
                let var = if count < 2 {
                    f64::NAN
                } else if first.is_finite() && all_equal {
                    0.0
                } else {
                    let mean = sum / count as f64;
                    let mut sbb = 0.0;
                    for &v in &ret_b.values[start..=j] {
                        if !v.is_nan() {
                            let db = v - mean;
                            sbb += db * db;
                        }
                    }
                    sbb / (count - 1) as f64
                };
                var_b.push(var);
            }

            let mut values = Vec::with_capacity(union_dates.len());
            let mut b_idx = 0;
            for idx in 0..union_dates.len() {
                while b_idx < ret_b.dates.len() && ret_b.dates[b_idx] < union_dates[idx] {
                    b_idx += 1;
                }
                let denom = if b_idx < ret_b.dates.len() && ret_b.dates[b_idx] == union_dates[idx] {
                    var_b[b_idx]
                } else {
                    f64::NAN
                };
                if denom == 0.0 || denom.is_nan() {
                    values.push(f64::NAN);
                    continue;
                }
                let start = (idx + 1).saturating_sub(win_size);
                let mut count = 0usize;
                let mut sum_x = 0.0;
                let mut sum_b = 0.0;
                for (&vx, &vb) in union_x[start..=idx].iter().zip(&union_b[start..=idx]) {
                    if !vx.is_nan() && !vb.is_nan() {
                        sum_x += vx;
                        sum_b += vb;
                        count += 1;
                    }
                }
                if count < 2 {
                    values.push(f64::NAN);
                } else {
                    let mean_x = sum_x / count as f64;
                    let mean_b = sum_b / count as f64;
                    let mut sxb = 0.0;
                    for (&vx, &vb) in union_x[start..=idx].iter().zip(&union_b[start..=idx]) {
                        if !vx.is_nan() && !vb.is_nan() {
                            sxb += (vx - mean_x) * (vb - mean_b);
                        }
                    }
                    let cov = sxb / (count - 1) as f64;
                    values.push(cov / denom);
                }
            }
            (union_dates, values)
        }
    };

    for v in res_values.iter_mut().take(3) {
        *v = f64::NAN;
    }

    let mut out_dates = Vec::new();
    let mut out_values = Vec::new();
    let mut k = 0;
    for (idx, &date) in x.dates.iter().enumerate() {
        while k < res_dates.len() && res_dates[k] < date {
            k += 1;
        }
        let value = if k < res_dates.len() && res_dates[k] == date {
            res_values[k]
        } else {
            f64::NAN
        };
        let include = match window {
            Window::Expanding => true,
            Window::Observations { ramp, .. } => idx >= ramp,
            Window::Days { ramp, .. } => date.ordinal() - x.dates[0].ordinal() >= i64::from(ramp),
        };
        if include {
            out_dates.push(date);
            out_values.push(value);
        }
    }
    Series::try_new(out_dates, out_values)
}

/// Rolling or expanding maximum peak-to-trough drawdown as a ratio.
///
/// Computes the rolling maximum price over the window, forms the drawdown ratio
/// `price / rolling_max - 1.0`, and then takes the rolling minimum of those ratios over the same
/// window before applying the ramp. For example, a 20% drawdown is returned as `-0.2`.
///
/// For `Window::Expanding` (size equal to `x.dates().len()`, ramp `0`) and `Window::Observations`,
/// both rolling stages skip `NaN` and `±Inf` (matching upstream pandas `rolling().max()` and
/// `rolling().min()`). For `Window::Days`, both stages use `(date - size, date]` and skip `NaN`
/// while retaining `±Inf` (matching upstream pandas `Series.max()` and `Series.min()`). Windows
/// with no eligible values and `0.0 / 0.0` ratios yield `NaN`.
///
/// Empty input returns empty for `Window::Expanding`, `Window::Observations`, and
/// `Window::Days { ramp: 0, .. }`, and returns `Err(Error::Ramp)` for `Window::Days` with
/// `ramp > 0` (mapping upstream's `IndexError` on an empty date index with a positive calendar
/// ramp).
pub fn max_drawdown(x: &Series, window: Window) -> Result<Series, Error> {
    if x.dates.is_empty() {
        if matches!(window, Window::Days { ramp, .. } if ramp > 0) {
            return Err(Error::Ramp);
        }
        return Series::try_new(vec![], vec![]);
    }
    match window {
        Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
            return Err(Error::Window)
        }
        Window::Observations { ramp, .. } if ramp > x.dates.len() => return Err(Error::Ramp),
        Window::Observations { size, .. } if size > x.dates.len() => {
            return Series::try_new(vec![], vec![])
        }
        _ => {}
    }

    let n = x.dates.len();
    let mut scores = Vec::with_capacity(n);
    let mut out_dates = Vec::new();
    let mut out_values = Vec::new();

    match window {
        Window::Days { size, ramp } => {
            let mut starts = Vec::with_capacity(n);
            let mut start = 0;
            for i in 0..n {
                while start < i
                    && x.dates[i].ordinal() - x.dates[start].ordinal() >= i64::from(size)
                {
                    start += 1;
                }
                starts.push(start);
                let mut peak: Option<f64> = None;
                for &v in &x.values[start..=i] {
                    if !v.is_nan() {
                        peak = Some(match peak {
                            Some(cur) if cur > v => cur,
                            _ => v,
                        });
                    }
                }
                let denom = peak.unwrap_or(f64::NAN);
                scores.push(x.values[i] / denom - 1.0);
            }
            for i in 0..n {
                if x.dates[i].ordinal() - x.dates[0].ordinal() < i64::from(ramp) {
                    continue;
                }
                let mut trough: Option<f64> = None;
                for &v in &scores[starts[i]..=i] {
                    if !v.is_nan() {
                        trough = Some(match trough {
                            Some(cur) => cur.min(v),
                            None => v,
                        });
                    }
                }
                out_dates.push(x.dates[i]);
                out_values.push(trough.unwrap_or(f64::NAN));
            }
        }
        Window::Expanding | Window::Observations { .. } => {
            let (win_size, ramp) = match window {
                Window::Expanding => (n, 0),
                Window::Observations { size, ramp } => (size, ramp),
                Window::Days { .. } => unreachable!(),
            };
            for i in 0..n {
                let start = (i + 1).saturating_sub(win_size);
                let mut peak: Option<f64> = None;
                for &v in &x.values[start..=i] {
                    if v.is_finite() {
                        peak = Some(match peak {
                            Some(cur) => cur.max(v),
                            None => v,
                        });
                    }
                }
                let denom = peak.unwrap_or(f64::NAN);
                scores.push(x.values[i] / denom - 1.0);
            }
            for i in ramp..n {
                let start = (i + 1).saturating_sub(win_size);
                let mut trough: Option<f64> = None;
                for &v in &scores[start..=i] {
                    if v.is_finite() {
                        trough = Some(match trough {
                            Some(cur) => cur.min(v),
                            None => v,
                        });
                    }
                }
                out_dates.push(x.dates[i]);
                out_values.push(trough.unwrap_or(f64::NAN));
            }
        }
    }

    Series::try_new(out_dates, out_values)
}

fn ewm_mean_com(values: &[f64], com: f64) -> Vec<f64> {
    let alpha = 1.0 / (1.0 + com);
    let old_wt_factor = 1.0 - alpha;
    let new_wt = alpha;

    let n = values.len();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }

    let mut weighted_avg = values[0];
    let mut has_obs = weighted_avg.is_finite();
    out.push(if has_obs { weighted_avg } else { f64::NAN });
    let mut old_wt = 1.0;

    for &cur in &values[1..] {
        let is_obs = cur.is_finite();
        if has_obs {
            old_wt *= old_wt_factor;
            if is_obs {
                if weighted_avg != cur {
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt);
                }
                old_wt = 1.0;
            }
        } else if is_obs {
            weighted_avg = cur;
            has_obs = true;
        }
        out.push(if has_obs { weighted_avg } else { f64::NAN });
    }

    out
}

fn step_fill(values: &mut [f64]) {
    let mut last: Option<f64> = None;
    for v in values.iter_mut() {
        if v.is_nan() {
            if let Some(prev) = last {
                *v = prev;
            }
        } else {
            last = Some(*v);
        }
    }
    let mut next: Option<f64> = None;
    for v in values.iter_mut().rev() {
        if v.is_nan() {
            if let Some(after) = next {
                *v = after;
            }
        } else {
            next = Some(*v);
        }
    }
}

/// Exponentially weighted moving average over observation order.
///
/// When `beta` is `None`, uses the upstream default `0.75`. Valid `beta` values satisfy
/// `0.0 <= beta < 1.0` (`-0.0` is accepted as `0.0`); negative values, values `>= 1.0`, `±Inf`,
/// and `NaN` return `Err(Error::Beta)` even on empty input. Upstream `gs-quant` documents
/// `0 <= beta < 1`, whereas pandas `get_center_of_mass` checks `alpha <= 0 or alpha > 1` and
/// allows `NaN` to slip past; Rust strictly enforces the documented `[0, 1)` domain.
///
/// Decay follows row order rather than calendar-day spacing, matching upstream
/// `x.ewm(alpha=1 - beta, adjust=False).mean()`. Leading `NaN` and `±Inf` values emit `NaN`
/// until the first finite observation. Subsequent `NaN` and `±Inf` values are treated as missing:
/// the previous average is carried forward while each missing row decays the prior weight when
/// the next finite observation arrives. When a finite observation equals the current average
/// (`weighted_avg == cur`, including `-0.0 == +0.0`), the average and its sign are preserved.
pub fn exponential_moving_average(x: &Series, beta: Option<f64>) -> Result<Series, Error> {
    let beta = beta.unwrap_or(0.75);
    if !(0.0..1.0).contains(&beta) {
        return Err(Error::Beta);
    }
    let alpha_in = 1.0 - beta;
    let com = (1.0 - alpha_in) / alpha_in;
    let values = ewm_mean_com(&x.values, com);
    Series::try_new(x.dates.clone(), values)
}

/// Moving Average Convergence Divergence (MACD) over observation order.
///
/// Mirrors upstream `gs_quant/timeseries/technicals.py:macd` (lines 299–344) and
/// `gs_quant/timeseries/algebra.py:subtract` (default `Interpolate.STEP` = `ffill().bfill()`):
///
/// 1. Defaults `m = 12`, `n = 26`, `s = 1` when `None`, and validates `m >= 1`, `n >= 1`,
///    `s >= 1` first—returning `Err(Error::Span)` if any span is `0`, even on empty input.
/// 2. Computes observation-order EWMs `a = ewm(x, com = (m - 1) / 2)` and
///    `b = ewm(x, com = (n - 1) / 2)` with `adjust = false`.
/// 3. Applies `ffill().bfill()` independently to `a` and `b` (matching `subtract(..., Interpolate.STEP)`).
///    When `x` has leading `NaN` or `±Inf` followed by a finite observation, both `a` and `b`
///    backfill those leading rows with that first finite value, so `diff = a_filled - b_filled`
///    starts with `+0.0` (which enters the third EWM when `s > 1`). If `x` has no finite values,
///    `a`, `b`, `diff`, and the result remain all `NaN`.
/// 4. Applies the third EWM `ewm(diff, com = (s - 1) / 2)` with `adjust = false` (treating any
///    `a - b` overflow to `±Inf` as missing in that EWM) and returns the result on `x.dates()`.
pub fn macd(
    x: &Series,
    m: Option<usize>,
    n: Option<usize>,
    s: Option<usize>,
) -> Result<Series, Error> {
    let m = m.unwrap_or(12);
    let n = n.unwrap_or(26);
    let s = s.unwrap_or(1);
    if m == 0 || n == 0 || s == 0 {
        return Err(Error::Span);
    }
    if x.dates.is_empty() {
        return Series::try_new(vec![], vec![]);
    }

    let mut a = ewm_mean_com(&x.values, (m - 1) as f64 / 2.0);
    let mut b = ewm_mean_com(&x.values, (n - 1) as f64 / 2.0);
    step_fill(&mut a);
    step_fill(&mut b);

    let diff: Vec<f64> = a.iter().zip(&b).map(|(&av, &bv)| av - bv).collect();
    let out = ewm_mean_com(&diff, (s - 1) as f64 / 2.0);
    Series::try_new(x.dates.clone(), out)
}

/// Smoothed (modified / running) moving average over a rolling or expanding window.
///
/// Mirrors upstream `gs_quant/timeseries/technicals.py:smoothed_moving_average` on a
/// `datetime.date` index:
///
/// 1. Normalizes `window` (`Window::Expanding` uses size `x.dates().len()` and ramp `0`).
/// 2. Computes the initial value as the first post-ramp entry of `mean(x, Window(size, 0))`,
///    skipping `NaN` and `±Inf` in that initial window (yielding `NaN` if none are finite, and
///    preserving `-0.0` when all finite values in that window are `-0.0`).
/// 3. Applies the ramp to `x` first, then iterates over the post-ramp rows with
///    `out[i] = ((n - 1) * out[i - 1] + x[i]) / n`. For `Window::Expanding` and
///    `Window::Observations`, `n` is the normalized window size; for `Window::Days`, `n` is the
///    number of post-ramp observations in `(date - size, date]`.
///
/// Empty input returns empty for `Window::Expanding`, `Window::Observations`, and
/// `Window::Days { ramp: 0, .. }`, and returns `Err(Error::Ramp)` for `Window::Days` with
/// `ramp > 0` (mapping upstream's `IndexError`). On non-empty input, `size == 0` returns
/// `Err(Error::Window)`, observation `ramp > x.dates().len()` returns `Err(Error::Ramp)`, and
/// observation `size > x.dates().len()` (or a ramp leaving no rows) returns an empty `Series`.
pub fn smoothed_moving_average(x: &Series, window: Window) -> Result<Series, Error> {
    if x.dates.is_empty() {
        if matches!(window, Window::Days { ramp, .. } if ramp > 0) {
            return Err(Error::Ramp);
        }
        return Series::try_new(vec![], vec![]);
    }
    let n = x.dates.len();
    match window {
        Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
            return Err(Error::Window)
        }
        Window::Observations { ramp, .. } if ramp > n => return Err(Error::Ramp),
        Window::Observations { size, .. } if size > n => return Series::try_new(vec![], vec![]),
        _ => {}
    }

    let ramp_idx = match window {
        Window::Expanding => 0,
        Window::Observations { ramp, .. } => {
            if ramp >= n {
                return Series::try_new(vec![], vec![]);
            }
            ramp
        }
        Window::Days { ramp, .. } => {
            match (0..n).find(|&i| x.dates[i].ordinal() - x.dates[0].ordinal() >= i64::from(ramp)) {
                Some(idx) => idx,
                None => return Series::try_new(vec![], vec![]),
            }
        }
    };

    let init_start = match window {
        Window::Expanding => 0,
        Window::Observations { size, .. } => (ramp_idx + 1).saturating_sub(size),
        Window::Days { size, .. } => (0..=ramp_idx)
            .find(|&j| x.dates[ramp_idx].ordinal() - x.dates[j].ordinal() < i64::from(size))
            .unwrap_or(ramp_idx),
    };
    let mut init_sum = -0.0_f64;
    let mut init_count = 0usize;
    for &v in &x.values[init_start..=ramp_idx] {
        if v.is_finite() {
            init_sum += v;
            init_count += 1;
        }
    }
    let initial_moving_average = if init_count == 0 {
        f64::NAN
    } else {
        init_sum / init_count as f64
    };

    let sliced_dates = &x.dates[ramp_idx..];
    let sliced_values = &x.values[ramp_idx..];
    let m = sliced_dates.len();
    let mut out_values = Vec::with_capacity(m);
    out_values.push(initial_moving_average);

    let mut start = 0usize;
    for i in 1..m {
        let window_num_elem = match window {
            Window::Expanding => n as f64,
            Window::Observations { size, .. } => size as f64,
            Window::Days { size, .. } => {
                while start < i
                    && sliced_dates[i].ordinal() - sliced_dates[start].ordinal() >= i64::from(size)
                {
                    start += 1;
                }
                (i + 1 - start) as f64
            }
        };
        let prev = out_values[i - 1];
        let cur = sliced_values[i];
        out_values.push(((window_num_elem - 1.0) * prev + cur) / window_num_elem);
    }

    Series::try_new(sliced_dates.to_vec(), out_values)
}

/// Relative Strength Index (RSI) on a `0.0` to `100.0` scale over a rolling window.
///
/// Mirrors upstream `gs_quant/timeseries/technicals.py:relative_strength_index` on a
/// `datetime.date` index:
///
/// 1. Defaults `window` to `Window::Observations { size: 14, ramp: 14 }` when `None`, and
///    normalizes `window` against the original series `x` (`Window::Expanding` becomes
///    `Window::Observations { size: x.dates().len(), ramp: 0 }`).
/// 2. Computes one-period price changes `diff(x, 1)[1:]` (length `x.dates().len().saturating_sub(1)`),
///    splitting positive changes into `gains` and negated negative changes into `losses`.
/// 3. Applies [`smoothed_moving_average`] with the normalized window to `gains` and `losses`.
/// 4. At each output date, sets RSI to `100.0` when `avg_losses == 0.0` (including constant-price
///    series where both `gains` and `losses` are zero), and otherwise computes
///    `100.0 - 100.0 / (1.0 + avg_gains / avg_losses)`.
///
/// Because `Window::Expanding` normalizes to `size = x.dates().len()` while `gains` and `losses`
/// have length `x.dates().len().saturating_sub(1)`, `Window::Expanding` always yields an empty
/// `Series`. Empty input returns empty for `None`, `Window::Expanding`, `Window::Observations`,
/// and `Window::Days { ramp: 0, .. }`, and returns `Err(Error::Ramp)` for `Window::Days` with
/// `ramp > 0` (mapping upstream's `IndexError`). On non-empty input, `size == 0` returns
/// `Err(Error::Window)` and observation `ramp > x.dates().len()` (or `ramp > gains.len()` when
/// `gains` is non-empty) returns `Err(Error::Ramp)`.
pub fn relative_strength_index(x: &Series, window: Option<Window>) -> Result<Series, Error> {
    let normalized = match window.unwrap_or(Window::Observations { size: 14, ramp: 14 }) {
        Window::Expanding => Window::Observations {
            size: x.dates.len(),
            ramp: 0,
        },
        w => w,
    };
    if !x.dates.is_empty() {
        match normalized {
            Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
                return Err(Error::Window)
            }
            Window::Observations { ramp, .. } if ramp > x.dates.len() => return Err(Error::Ramp),
            _ => {}
        }
    }

    let (gains, losses) = if x.dates.len() <= 1 {
        (
            Series::try_new(vec![], vec![])?,
            Series::try_new(vec![], vec![])?,
        )
    } else {
        let dates = x.dates[1..].to_vec();
        let mut gain_values = Vec::with_capacity(dates.len());
        let mut loss_values = Vec::with_capacity(dates.len());
        for pair in x.values.windows(2) {
            let change = pair[1] - pair[0];
            let gain = if change < 0.0 { 0.0 } else { change };
            let loss = if change > 0.0 {
                0.0
            } else if change < 0.0 {
                -change
            } else {
                change
            };
            gain_values.push(gain);
            loss_values.push(loss);
        }
        (
            Series::try_new(dates.clone(), gain_values)?,
            Series::try_new(dates, loss_values)?,
        )
    };

    let moving_avg_gains = smoothed_moving_average(&gains, normalized)?;
    let moving_avg_losses = smoothed_moving_average(&losses, normalized)?;

    let rsi_values = moving_avg_gains
        .values
        .iter()
        .zip(&moving_avg_losses.values)
        .map(|(&avg_gain, &avg_loss)| {
            if avg_loss == 0.0 {
                100.0
            } else {
                let relative_strength = avg_gain / avg_loss;
                100.0 - (100.0 / (1.0 + relative_strength))
            }
        })
        .collect();

    Series::try_new(moving_avg_gains.dates, rsi_values)
}

/// Unbiased sample standard deviation (`ddof = 1`) over a rolling or expanding window.
///
/// Mirrors upstream `gs_quant/timeseries/statistics.py:std` and `rolling_std` (lines 55–70,
/// 535–580) on a `datetime.date` index:
///
/// - Empty input returns an empty `Series` before window or ramp validation.
/// - On non-empty input, `size == 0` returns `Err(Error::Window)` (including
///   `Window::Days { size: 0, .. }`, whereas upstream `DateOffset(days=0)` silently falls through
///   as an expanding window), observation `ramp > x.dates().len()` returns `Err(Error::Ramp)`, and
///   observation `size > x.dates().len()` or a ramp leaving no rows returns an empty `Series`.
/// - For `Window::Expanding` and `Window::Observations`, each window aggregates finite values
///   (`v.is_finite()`, skipping both `NaN` and `±Inf`, matching pandas `rolling().std()`).
/// - For `Window::Days`, each half-open calendar window `(date - size, date]` aggregates non-`NaN`
///   values (`!v.is_nan()`, matching NumPy `np.std(section[section == section], ddof=1)`): if any
///   `±Inf` is present in the window, emits `NaN`.
/// - When the aggregated count is `< 2`, emits `f64::NAN`. When all aggregated finite values are
///   equal (including `4.2`, `0.1`, or `-0.0`), emits `+0.0`. Otherwise computes two-pass unbiased
///   sample standard deviation `sqrt(sum((v - mean)^2) / (count - 1))`.
pub fn std(x: &Series, window: Window) -> Result<Series, Error> {
    if x.dates.is_empty() {
        return Series::try_new(vec![], vec![]);
    }
    let n = x.dates.len();
    match window {
        Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
            return Err(Error::Window)
        }
        Window::Observations { ramp, .. } if ramp > n => return Err(Error::Ramp),
        Window::Observations { size, .. } if size > n => return Series::try_new(vec![], vec![]),
        Window::Observations { ramp, .. } if ramp >= n => return Series::try_new(vec![], vec![]),
        _ => {}
    }

    let mut dates = Vec::new();
    let mut values = Vec::new();
    let mut day_start = 0usize;

    for i in 0..n {
        let include = match window {
            Window::Expanding => true,
            Window::Observations { ramp, .. } => i >= ramp,
            Window::Days { ramp, .. } => {
                x.dates[i].ordinal() - x.dates[0].ordinal() >= i64::from(ramp)
            }
        };
        if !include {
            continue;
        }
        let start = match window {
            Window::Expanding => 0,
            Window::Observations { size, .. } => (i + 1).saturating_sub(size),
            Window::Days { size, .. } => {
                while day_start < i
                    && x.dates[i].ordinal() - x.dates[day_start].ordinal() >= i64::from(size)
                {
                    day_start += 1;
                }
                day_start
            }
        };

        let slice = &x.values[start..=i];
        let val = match window {
            Window::Expanding | Window::Observations { .. } => {
                let mut sum = 0.0;
                let mut count = 0usize;
                let mut first = 0.0;
                let mut all_equal = true;
                for &v in slice {
                    if v.is_finite() {
                        if count == 0 {
                            first = v;
                        } else if v != first {
                            all_equal = false;
                        }
                        sum += v;
                        count += 1;
                    }
                }
                if count < 2 {
                    f64::NAN
                } else if all_equal {
                    0.0
                } else {
                    let mean = sum / count as f64;
                    let mut sum_sq = 0.0;
                    for &v in slice {
                        if v.is_finite() {
                            let d = v - mean;
                            sum_sq += d * d;
                        }
                    }
                    (sum_sq / (count - 1) as f64).sqrt()
                }
            }
            Window::Days { .. } => {
                let mut sum = 0.0;
                let mut count = 0usize;
                let mut first = 0.0;
                let mut all_equal = true;
                let mut has_inf = false;
                for &v in slice {
                    if !v.is_nan() {
                        if !v.is_finite() {
                            has_inf = true;
                        } else if count == 0 {
                            first = v;
                        } else if v != first {
                            all_equal = false;
                        }
                        sum += v;
                        count += 1;
                    }
                }
                if count < 2 || has_inf {
                    f64::NAN
                } else if all_equal {
                    0.0
                } else {
                    let mean = sum / count as f64;
                    let mut sum_sq = 0.0;
                    for &v in slice {
                        if !v.is_nan() {
                            let d = v - mean;
                            sum_sq += d * d;
                        }
                    }
                    (sum_sq / (count - 1) as f64).sqrt()
                }
            }
        };

        dates.push(x.dates[i]);
        values.push(val);
    }

    Series::try_new(dates, values)
}

/// Lower and upper Bollinger bands sharing identical dates.
#[derive(Debug, Clone, PartialEq)]
pub struct BollingerBands {
    pub lower: Series,
    pub upper: Series,
}

fn bollinger_mean(x: &Series, window: Window) -> Vec<f64> {
    let n = x.dates.len();
    let mut values = Vec::new();
    let mut day_start = 0usize;

    for i in 0..n {
        let include = match window {
            Window::Expanding => true,
            Window::Observations { ramp, .. } => i >= ramp,
            Window::Days { ramp, .. } => {
                x.dates[i].ordinal() - x.dates[0].ordinal() >= i64::from(ramp)
            }
        };
        if !include {
            continue;
        }
        let start = match window {
            Window::Expanding => 0,
            Window::Observations { size, .. } => (i + 1).saturating_sub(size),
            Window::Days { size, .. } => {
                while day_start < i
                    && x.dates[i].ordinal() - x.dates[day_start].ordinal() >= i64::from(size)
                {
                    day_start += 1;
                }
                day_start
            }
        };

        let slice = &x.values[start..=i];
        let mut sum = -0.0_f64;
        let mut count = 0usize;
        match window {
            Window::Expanding | Window::Observations { .. } => {
                for &v in slice {
                    if v.is_finite() {
                        sum += v;
                        count += 1;
                    }
                }
            }
            Window::Days { .. } => {
                for &v in slice {
                    if !v.is_nan() {
                        sum += v;
                        count += 1;
                    }
                }
            }
        }
        values.push(if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        });
    }

    values
}

/// Bollinger bands (`avg - k * sigma`, `avg + k * sigma`) over a rolling or expanding window.
///
/// Mirrors upstream `gs_quant/timeseries/technicals.py:bollinger_bands` (lines 98–143) on a
/// `datetime.date` index:
///
/// - Defaults `k = 2.0` when `None`. Any `f64` value of `k` (including `0.0`, `-0.0`, negative
///   values that swap `lower` and `upper`, `NaN`, and `±Inf`) follows IEEE-754 arithmetic without
///   validation errors.
/// - Empty input returns empty `lower` and `upper` `Series` before window or ramp validation.
/// - Reuses [`std`] for window/ramp validation and unbiased sample standard deviation (`ddof = 1`):
///   on non-empty input, `size == 0` returns `Err(Error::Window)` (including
///   `Window::Days { size: 0, .. }`, whereas upstream `DateOffset(days=0)` silently falls through
///   as an expanding window), observation `ramp > x.dates().len()` returns `Err(Error::Ramp)`, and
///   oversize observation windows or ramps leaving no rows return empty bands.
/// - Computes the rolling mean internally without altering [`moving_average`]: `Window::Expanding`
///   and `Window::Observations` aggregate finite values (`v.is_finite()`, skipping both `NaN` and
///   `±Inf`), whereas `Window::Days` aggregates non-`NaN` values (`!v.is_nan()`, so `±Inf` yields
///   `NaN` via `sigma`). Initializing the mean sum to `-0.0` preserves the upstream signed zero on
///   all-`-0.0` constant windows: `lower = -0.0`, `upper = +0.0` for finite `k > 0.0` or `+0.0`
///   (`k.is_sign_positive()`), and swapped to `lower = +0.0`, `upper = -0.0` for finite `k < 0.0`
///   or `-0.0` (`k.is_sign_negative()`).
/// - Documented upstream parity boundaries:
///   - Mixed signed zeros (`[+0.0, -0.0]`): under `Window::Expanding` and `Window::Observations`,
///     Rust's IEEE-754 summation (`+0.0 + (-0.0) = +0.0`) produces `(+0.0, +0.0)` (numeric
///     difference `0`), whereas pinned pandas 2.3.3 `rolling().mean()` takes the GH#42064
///     equal-value fast path and adopts the last zero's sign `-0.0`, producing `(-0.0, +0.0)`.
///   - Constant calendar-day windows with `k = ±Inf` (e.g. ten days of `4.2` under
///     `Window::Days { size: 10, ramp: 0 }`): Rust reuses [`std`]'s exact `+0.0` constant-window
///     contract (consistent with `std`, `correlation`, `beta`, and pandas observation windows), so
///     `0.0 * ±Inf = NaN`, whereas pinned NumPy 2.2.6 `rolling_std` retains a mean-rounding
///     residue (`~9.36e-16` on ten `4.2` values) that multiplies with `±Inf` into `(-Inf, +Inf)` /
///     `(+Inf, -Inf)`.
pub fn bollinger_bands(
    x: &Series,
    window: Window,
    k: Option<f64>,
) -> Result<BollingerBands, Error> {
    if x.dates.is_empty() {
        return Ok(BollingerBands {
            lower: Series::try_new(vec![], vec![])?,
            upper: Series::try_new(vec![], vec![])?,
        });
    }

    let sigma = std(x, window)?;
    if sigma.dates.is_empty() {
        return Ok(BollingerBands {
            lower: Series::try_new(vec![], vec![])?,
            upper: Series::try_new(vec![], vec![])?,
        });
    }

    let avg = bollinger_mean(x, window);
    let k = k.unwrap_or(2.0);
    let mut lower_values = Vec::with_capacity(sigma.values.len());
    let mut upper_values = Vec::with_capacity(sigma.values.len());
    for (&mean, &sd) in avg.iter().zip(&sigma.values) {
        let delta = k * sd;
        lower_values.push(mean - delta);
        upper_values.push(mean + delta);
    }

    Ok(BollingerBands {
        lower: Series::try_new(sigma.dates.clone(), lower_values)?,
        upper: Series::try_new(sigma.dates, upper_values)?,
    })
}

fn sample_cov_finite_pairs(xs: &[f64], ys: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut first_x = 0.0_f64;
    let mut first_y = 0.0_f64;
    let mut all_eq_x = true;
    let mut all_eq_y = true;

    for (&vx, &vy) in xs.iter().zip(ys) {
        if vx.is_finite() && vy.is_finite() {
            if count == 0 {
                first_x = vx;
                first_y = vy;
            } else {
                if vx != first_x {
                    all_eq_x = false;
                }
                if vy != first_y {
                    all_eq_y = false;
                }
            }
            sum_x += vx;
            sum_y += vy;
            count += 1;
        }
    }

    if count < 2 {
        return f64::NAN;
    }
    if all_eq_x || all_eq_y {
        return 0.0;
    }

    let count_f = count as f64;
    let mean_x = if sum_x.is_finite() {
        sum_x / count_f
    } else {
        xs.iter()
            .zip(ys)
            .filter(|&(&vx, &vy)| vx.is_finite() && vy.is_finite())
            .map(|(&vx, _)| vx / count_f)
            .sum()
    };
    let mean_y = if sum_y.is_finite() {
        sum_y / count_f
    } else {
        xs.iter()
            .zip(ys)
            .filter(|&(&vx, &vy)| vx.is_finite() && vy.is_finite())
            .map(|(_, &vy)| vy / count_f)
            .sum()
    };

    let mut sxy = 0.0_f64;
    for (&vx, &vy) in xs.iter().zip(ys) {
        if vx.is_finite() && vy.is_finite() {
            sxy += (vx - mean_x) * (vy - mean_y);
        }
    }
    sxy / (count - 1) as f64
}

fn sample_cov_non_nan_pairs(xs: &[f64], ys: &[f64]) -> f64 {
    let count = xs.len();
    if count < 2 {
        return f64::NAN;
    }

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let first_x = xs[0];
    let first_y = ys[0];
    let mut all_eq_x = true;
    let mut all_eq_y = true;

    for (&vx, &vy) in xs.iter().zip(ys) {
        if !vx.is_finite() || !vy.is_finite() {
            return f64::NAN;
        }
        if vx != first_x {
            all_eq_x = false;
        }
        if vy != first_y {
            all_eq_y = false;
        }
        sum_x += vx;
        sum_y += vy;
    }

    if all_eq_x || all_eq_y {
        return 0.0;
    }

    let count_f = count as f64;
    let mean_x = if sum_x.is_finite() {
        sum_x / count_f
    } else {
        xs.iter().map(|&vx| vx / count_f).sum()
    };
    let mean_y = if sum_y.is_finite() {
        sum_y / count_f
    } else {
        ys.iter().map(|&vy| vy / count_f).sum()
    };

    let mut sxy = 0.0_f64;
    for (&vx, &vy) in xs.iter().zip(ys) {
        sxy += (vx - mean_x) * (vy - mean_y);
    }
    sxy / (count - 1) as f64
}

/// Unbiased rolling or expanding sample covariance (`ddof = 1`) between two date-indexed series.
///
/// Mirrors upstream `gs_quant/timeseries/statistics.py:cov` (lines 678–724) and
/// `gs_quant/timeseries/helper.py:{normalize_window,apply_ramp}` on a `datetime.date` index:
///
/// - **Validation:**
///   - When `x` is non-empty, `size == 0` returns `Err(Error::Window)` (for both
///     `Window::Observations` and `Window::Days`) and observation `ramp > x.dates().len()`
///     returns `Err(Error::Ramp)`.
///   - Under `Window::Days`, an empty `x` returns an empty `Series`.
///   - Under `Window::Expanding` and `Window::Observations`, the sorted date union of `x` and `y`
///     is formed first (filling unmatched dates with `NaN`), with effective `(win_size, ramp)`
///     equal to `(x.dates().len(), 0)` for `Window::Expanding` and `(size, ramp)` for
///     `Window::Observations`. If the union is empty, returns empty; if `win_size == 0` (e.g.
///     empty `x` with non-empty `y` under `Window::Expanding` or `size == 0`), returns
///     `Err(Error::Window)`; if `ramp > union.len()`, returns `Err(Error::Ramp)`; if
///     `win_size > union.len()` or `ramp >= union.len()`, returns an empty `Series`.
/// - **Window computation:**
///   - `Window::Expanding` and `Window::Observations` emit `union_dates[ramp..]` and aggregate
///     pairs in `(idx + 1).saturating_sub(win_size)..=idx` where `vx.is_finite() && vy.is_finite()`
///     (skipping both `NaN` and `±Inf`, matching pandas 2.3.3 `rolling().cov()`).
///   - `Window::Days` emits `x.dates()` at or after `x.dates()[0] + ramp` and aggregates dates in
///     `(date - size, date]` present in both `x` and `y` where `!vx.is_nan() && !vy.is_nan()`
///     (matching upstream `Series.cov(y)` / `np.cov`, where `±Inf` paired with `NaN` or an
///     unmatched date is dropped first, whereas `±Inf` paired with any non-`NaN` value yields
///     `NaN`).
///   - Windows with `< 2` kept pairs yield `f64::NAN`. Constant finite `x` or constant finite `y`
///     windows return exact `+0.0`. Otherwise computes two-pass centered sample covariance
///     `sum((vx - mean_x) * (vy - mean_y)) / (count - 1)` with an overflow-safe `sum(v / count)`
///     mean fallback when the running sum overflows `f64::MAX`.
/// - **Documented upstream parity boundaries:**
///   - `Window::Days { size: 0, .. }` on non-empty `x` returns `Err(Error::Window)` consistently
///     across the crate, whereas upstream `DateOffset(days=0)` produces all `NaN`.
///   - Empty `x` with `Window::Days { size > 0, ramp > 0 }` returns an empty `Series`, whereas
///     upstream `apply_ramp` raises `IndexError` indexing an empty date index.
///   - Constant finite windows return exact `+0.0`, whereas upstream pandas/NumPy leave floating-point
///     rounding residue (e.g. `-2.66e-15` in `rolling().cov()` on `4.2`, `2.89e-34` in `np.cov`
///     on `0.1`).
///   - Large-offset observation/expanding windows (e.g. `1e9 + [1, 2, 4, 7, 11]` vs
///     `1e9 + [2, 5, 3, 9, 12]`) match the exact mathematical covariance (and upstream
///     `Window::Days` / `np.cov`) via two-pass centering, whereas pinned pandas 2.3.3
///     `rolling().cov()` suffers catastrophic cancellation from single-pass `E[XY] - E[X]E[Y]`.
///   - [`Series::try_new`] enforces strictly increasing unique dates on both `x` and `y` (upstream
///     only asserts monotonicity on `x`).
pub fn cov(x: &Series, y: &Series, window: Window) -> Result<Series, Error> {
    if !x.dates.is_empty() {
        match window {
            Window::Observations { size: 0, .. } | Window::Days { size: 0, .. } => {
                return Err(Error::Window);
            }
            Window::Observations { ramp, .. } if ramp > x.dates.len() => {
                return Err(Error::Ramp);
            }
            _ => {}
        }
    }

    match window {
        Window::Days { size, ramp } => {
            if x.dates.is_empty() {
                return Series::try_new(vec![], vec![]);
            }

            let cap = x.dates.len().min(y.dates.len());
            let mut common_dates = Vec::with_capacity(cap);
            let mut common_x = Vec::with_capacity(cap);
            let mut common_y = Vec::with_capacity(cap);
            let mut i = 0;
            let mut j = 0;
            while i < x.dates.len() && j < y.dates.len() {
                match x.dates[i].cmp(&y.dates[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        let vx = x.values[i];
                        let vy = y.values[j];
                        if !vx.is_nan() && !vy.is_nan() {
                            common_dates.push(x.dates[i]);
                            common_x.push(vx);
                            common_y.push(vy);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }

            let first_ord = x.dates[0].ordinal();
            let size_i64 = i64::from(size);
            let ramp_i64 = i64::from(ramp);
            let mut out_dates = Vec::new();
            let mut out_values = Vec::new();
            let mut lo = 0usize;
            let mut hi = 0usize;

            for &d in &x.dates {
                let d_ord = d.ordinal();
                if d_ord - first_ord < ramp_i64 {
                    continue;
                }
                while hi < common_dates.len() && common_dates[hi] <= d {
                    hi += 1;
                }
                while lo < hi && d_ord - common_dates[lo].ordinal() >= size_i64 {
                    lo += 1;
                }
                out_dates.push(d);
                out_values.push(sample_cov_non_nan_pairs(
                    &common_x[lo..hi],
                    &common_y[lo..hi],
                ));
            }

            Series::try_new(out_dates, out_values)
        }
        Window::Expanding | Window::Observations { .. } => {
            let cap = x.dates.len() + y.dates.len();
            let mut union_dates = Vec::with_capacity(cap);
            let mut union_x = Vec::with_capacity(cap);
            let mut union_y = Vec::with_capacity(cap);
            let mut i = 0;
            let mut j = 0;
            while i < x.dates.len() || j < y.dates.len() {
                let ord = if i < x.dates.len() && j < y.dates.len() {
                    x.dates[i].cmp(&y.dates[j])
                } else if i < x.dates.len() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
                match ord {
                    std::cmp::Ordering::Less => {
                        union_dates.push(x.dates[i]);
                        union_x.push(x.values[i]);
                        union_y.push(f64::NAN);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        union_dates.push(y.dates[j]);
                        union_x.push(f64::NAN);
                        union_y.push(y.values[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        union_dates.push(x.dates[i]);
                        union_x.push(x.values[i]);
                        union_y.push(y.values[j]);
                        i += 1;
                        j += 1;
                    }
                }
            }

            let (win_size, ramp) = match window {
                Window::Expanding => (x.dates.len(), 0),
                Window::Observations { size, ramp } => (size, ramp),
                Window::Days { .. } => unreachable!(),
            };

            let n = union_dates.len();
            if n == 0 {
                return Series::try_new(vec![], vec![]);
            }
            if win_size == 0 {
                return Err(Error::Window);
            }
            if ramp > n {
                return Err(Error::Ramp);
            }
            if win_size > n || ramp >= n {
                return Series::try_new(vec![], vec![]);
            }

            let mut out_dates = Vec::with_capacity(n - ramp);
            let mut out_values = Vec::with_capacity(n - ramp);
            for idx in ramp..n {
                let start = (idx + 1).saturating_sub(win_size);
                out_dates.push(union_dates[idx]);
                out_values.push(sample_cov_finite_pairs(
                    &union_x[start..=idx],
                    &union_y[start..=idx],
                ));
            }

            Series::try_new(out_dates, out_values)
        }
    }
}

fn vec_norm2(xs: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    for &x in xs {
        let a = x.abs();
        if a > scale {
            scale = a;
        }
    }
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let mut sum_sq = 0.0_f64;
    for &x in xs {
        let r = x / scale;
        sum_sq += r * r;
    }
    scale * sum_sq.sqrt()
}

/// Solves `min_beta ||A * beta - b||_2` via Householder QR with column pivoting.
/// Returns `None` if `A` is rank-deficient or produces non-finite coefficients.
fn solve_window_qr(x_win: &[Vec<f64>], y_win: &[f64], k: usize) -> Option<Vec<f64>> {
    let w = y_win.len();
    let mut cols: Vec<Vec<f64>> = (0..k)
        .map(|c| x_win.iter().map(|row| row[c]).collect())
        .collect();
    let mut qtb = y_win.to_vec();
    let mut perm: Vec<usize> = (0..k).collect();
    let mut v = vec![0.0_f64; w];

    for j in 0..k {
        let mut piv = j;
        let mut best_norm = vec_norm2(&cols[j][j..w]);
        for (c, col) in cols.iter().enumerate().skip(j + 1) {
            let n = vec_norm2(&col[j..w]);
            if n > best_norm {
                best_norm = n;
                piv = c;
            }
        }
        if piv != j {
            cols.swap(j, piv);
            perm.swap(j, piv);
        }
        if !best_norm.is_finite() || best_norm == 0.0 {
            return None;
        }

        let x0 = cols[j][j];
        let r_jj = if x0 >= 0.0 { -best_norm } else { best_norm };
        let v0 = x0 - r_jj;
        let len = w - j;
        v[0] = 1.0;
        let mut v_norm_sq = 1.0_f64;
        for (vm_out, &xj) in v[1..len].iter_mut().zip(&cols[j][j + 1..w]) {
            let vm = xj / v0;
            *vm_out = vm;
            v_norm_sq += vm * vm;
        }
        let beta_h = 2.0 / v_norm_sq;
        cols[j][j] = r_jj;

        for col in cols.iter_mut().skip(j + 1) {
            let mut dot = col[j];
            for (&vm, &cm) in v[1..len].iter().zip(&col[j + 1..w]) {
                dot += vm * cm;
            }
            let factor = beta_h * dot;
            col[j] -= factor;
            for (&vm, cm) in v[1..len].iter().zip(&mut col[j + 1..w]) {
                *cm -= factor * vm;
            }
        }

        let mut dot_b = qtb[j];
        for (&vm, &bm) in v[1..len].iter().zip(&qtb[j + 1..w]) {
            dot_b += vm * bm;
        }
        let factor_b = beta_h * dot_b;
        qtb[j] -= factor_b;
        for (&vm, bm) in v[1..len].iter().zip(&mut qtb[j + 1..w]) {
            *bm -= factor_b * vm;
        }
    }

    let r00 = cols[0][0].abs();
    if !r00.is_finite() || r00 == 0.0 {
        return None;
    }
    let tol = (w.max(k) as f64) * f64::EPSILON * r00;
    for (j, col) in cols.iter().enumerate() {
        let rjj = col[j].abs();
        if !rjj.is_finite() || rjj <= tol {
            return None;
        }
    }

    let mut z = vec![0.0_f64; k];
    for j in (0..k).rev() {
        let mut rhs = qtb[j];
        for c in (j + 1)..k {
            rhs -= cols[c][j] * z[c];
        }
        let zj = rhs / cols[j][j];
        if !zj.is_finite() {
            return None;
        }
        z[j] = zj;
    }

    let mut beta = vec![0.0_f64; k];
    for j in 0..k {
        beta[perm[j]] = z[j];
    }
    Some(beta)
}

/// Offline multi-factor rolling ordinary least squares regression.
///
/// Mirrors upstream `gs_quant/timeseries/statistics.py:RollingLinearRegression` (lines 1173–1257)
/// on a `datetime.date` index, solving each rolling window via pure-Rust Householder QR
/// decomposition with column pivoting:
///
/// - **Predictor alignment and intercept (`sm.add_constant(..., has_constant='skip')`):**
///   - Rejects an empty `predictors` slice with `Err(Error::Length)`.
///   - Outer-joins all `predictors` onto the sorted union of their dates (`pred_union_dates`),
///     filling missing dates with `f64::NAN`.
///   - When `fit_intercept` is `true`, prepends a constant `1.0` column unless any predictor column
///     on `pred_union_dates` is already globally non-zero constant without `NaN` or `±Inf`
///     (`!pred_union_dates.is_empty()`, every entry finite, non-zero, and equal to the first entry).
///     Output columns are indexed `0..k_cols`.
///   - When `fit_intercept` is `false`, no constant column is prepended and output columns are
///     indexed `1..=k_cols`.
/// - **Window validation and inner alignment with `y`:**
///   - Requires `window > k_cols` (validated before alignment), returning `Err(Error::Window)`
///     otherwise.
///   - Filters out rows where any design-matrix entry or `y` is non-finite (`NaN` or `±Inf`) and
///     inner-joins with `y`.
///   - Requires `aligned_dates.len() >= window`, returning `Err(Error::Window)` otherwise.
/// - **Per-window OLS solve and outputs:**
///   - The first `window - 1` aligned dates emit `f64::NAN` across all output series.
///   - Each trailing window of length `window` is factored via Householder QR with column pivoting.
///     If `|R[0][0]| == 0`, non-finite, or any diagonal `|R[j][j]| <= max(window, k_cols) * f64::EPSILON * |R[0][0]|`
///     (or any back-substituted coefficient is non-finite), the window is rank-deficient and emits
///     `f64::NAN` across all outputs.
///   - `r_squared` uses centered `SST` (`sum((y - mean_y)^2)`) when any column of the aligned
///     design matrix is globally non-zero constant (`has_constant`), and uncentered `SST`
///     (`sum(y^2)`) otherwise, matching `statsmodels.RollingOLS`. When `SST == 0.0`, `r_squared` is
///     `f64::NAN`.
///   - `fitted_values` evaluates `dot(X_aligned[idx], beta)` at each window's right endpoint, and
///     `standard_deviation_of_errors` evaluates `sqrt(SSE / (window - k_cols))`.
/// - **Documented upstream parity boundaries:**
///   - **Constant `y` with intercept (`SST == 0`):** Rust returns `r_squared = NaN` and exact
///     `std_err = +0.0` (exact `0 / 0`), whereas pinned `statsmodels 0.15.0` normal-equation
///     residual noise (`SSE ~ 1e-29`) can yield `r_squared = -Inf` and `std_err ~ 1e-15..1e-14`.
///   - **Ill-conditioned predictors (e.g. `1e7 + [1..6]` with intercept):** Rust Householder QR
///     avoids squaring the condition number, recovering accurate coefficients, `r_squared`, and
///     `std_err`, whereas pinned `statsmodels 0.15.0` normal equations `(X'X)^-1 X'y` lose ~5 digits
///     (e.g. `-19884102.88` vs true `-19999999.0`).
///   - **Singular / exact-collinear windows (rank-deficient `X`):** Rust Householder QR with
///     column pivoting consistently detects rank deficiency
///     (`|R[j][j]| <= max(window, k_cols) * f64::EPSILON * |R[0][0]|`) and emits `NaN` across
///     every singular window and output, whereas pinned `statsmodels 0.15.0` incremental `X'X`
///     normal-equation updates can leak floating-point roundoff on an exact rank-deficient window
///     and emit a spurious finite solve (e.g. `coef[0] = 16.0`, `r_squared = -1.24295` on
///     `[1, x_quad, 5]`, or `coef = [1.75, -9.953125, 4.7578125]`, `r_squared = -0.82570` on
///     `[1, x_quad, 2 * x_quad + 3]`).
///   - **Short or empty aligned series (`aligned.len() < window`):** Rust returns typed
///     `Err(Error::Window)` vs upstream `IndexError` / `ValueError`.
///   - **Empty predictor slice (`predictors.is_empty()`):** Rust returns typed `Err(Error::Length)`
///     vs upstream `ValueError: No objects to concatenate`.
///   - **Strictly increasing dates:** [`Series::try_new`] enforces strictly increasing unique dates
///     on all input series.
#[derive(Debug, Clone, PartialEq)]
pub struct RollingLinearRegression {
    column_indices: Vec<usize>,
    coefficients: Vec<Series>,
    r_squared: Series,
    fitted_values: Series,
    standard_deviation_of_errors: Series,
}

impl RollingLinearRegression {
    /// Fit a multi-factor rolling ordinary least squares regression over aligned finite rows.
    pub fn fit(
        predictors: &[Series],
        y: &Series,
        window: usize,
        fit_intercept: bool,
    ) -> Result<Self, Error> {
        if predictors.is_empty() {
            return Err(Error::Length);
        }

        let total_pred_dates: usize = predictors.iter().map(|s| s.dates.len()).sum();
        let mut pred_union_dates = Vec::with_capacity(total_pred_dates);
        for s in predictors {
            pred_union_dates.extend_from_slice(&s.dates);
        }
        pred_union_dates.sort_unstable();
        pred_union_dates.dedup();

        let n_pred = pred_union_dates.len();
        let mut pred_cols: Vec<Vec<f64>> = Vec::with_capacity(predictors.len());
        for s in predictors {
            let mut col = vec![f64::NAN; n_pred];
            let mut j = 0usize;
            for (i, &d) in pred_union_dates.iter().enumerate() {
                if j < s.dates.len() && s.dates[j] == d {
                    col[i] = s.values[j];
                    j += 1;
                }
            }
            pred_cols.push(col);
        }

        let is_nonzero_const = |col: &[f64]| {
            !col.is_empty()
                && col[0].is_finite()
                && col[0] != 0.0
                && col[1..].iter().all(|&v| v == col[0])
        };
        let prepend_const = fit_intercept && !pred_cols.iter().any(|c| is_nonzero_const(c));

        let mut design_cols = Vec::with_capacity(predictors.len() + usize::from(prepend_const));
        if prepend_const {
            design_cols.push(vec![1.0_f64; n_pred]);
        }
        design_cols.extend(pred_cols);

        let k_cols = design_cols.len();
        let column_indices: Vec<usize> = if fit_intercept {
            (0..k_cols).collect()
        } else {
            (1..=k_cols).collect()
        };

        if window <= k_cols {
            return Err(Error::Window);
        }

        let mut aligned_dates = Vec::new();
        let mut aligned_x: Vec<Vec<f64>> = Vec::new();
        let mut aligned_y = Vec::new();
        let mut i = 0usize;
        let mut j = 0usize;
        while i < pred_union_dates.len() && j < y.dates.len() {
            match pred_union_dates[i].cmp(&y.dates[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    let yv = y.values[j];
                    if yv.is_finite() && design_cols.iter().all(|col| col[i].is_finite()) {
                        aligned_dates.push(pred_union_dates[i]);
                        let mut row = Vec::with_capacity(k_cols);
                        for col in &design_cols {
                            row.push(col[i]);
                        }
                        aligned_x.push(row);
                        aligned_y.push(yv);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        let n_aligned = aligned_dates.len();
        if n_aligned < window {
            return Err(Error::Window);
        }

        let const_col_idx = (0..k_cols).find(|&c| {
            let v0 = aligned_x[0][c];
            v0 != 0.0 && aligned_x[1..].iter().all(|row| row[c] == v0)
        });
        let has_constant = const_col_idx.is_some();

        let mut coeff_vals: Vec<Vec<f64>> = vec![vec![f64::NAN; n_aligned]; k_cols];
        let mut r_sq_vals = vec![f64::NAN; n_aligned];
        let mut fitted_vals = vec![f64::NAN; n_aligned];
        let mut std_err_vals = vec![f64::NAN; n_aligned];
        let dof = (window - k_cols) as f64;

        for idx in (window - 1)..n_aligned {
            let win_start = idx + 1 - window;
            let x_win = &aligned_x[win_start..=idx];
            let y_win = &aligned_y[win_start..=idx];

            if let Some(beta) = solve_window_qr(x_win, y_win, k_cols) {
                let y_is_const = y_win[1..].iter().all(|&v| v == y_win[0]);
                if let Some(c_const) = const_col_idx {
                    if y_is_const {
                        let y0 = y_win[0];
                        let v0 = aligned_x[0][c_const];
                        let b_const = y0 / v0;
                        if b_const.is_finite() {
                            for (c, col_out) in coeff_vals.iter_mut().enumerate() {
                                col_out[idx] = if c == c_const { b_const } else { 0.0 };
                            }
                            fitted_vals[idx] = y0;
                            r_sq_vals[idx] = f64::NAN;
                            std_err_vals[idx] = 0.0;
                        }
                        continue;
                    }
                }

                let mut fitted = 0.0_f64;
                for (c, &b) in beta.iter().enumerate() {
                    fitted += aligned_x[idx][c] * b;
                }

                let mut sse = 0.0_f64;
                for (row, &yv) in x_win.iter().zip(y_win) {
                    let mut yhat = 0.0_f64;
                    for (c, &b) in beta.iter().enumerate() {
                        yhat += row[c] * b;
                    }
                    let resid = yv - yhat;
                    sse += resid * resid;
                }
                let std_err = (sse / dof).sqrt();

                let sst = if has_constant {
                    if y_is_const {
                        0.0
                    } else {
                        let sum_y: f64 = y_win.iter().copied().sum();
                        let mean_y = if sum_y.is_finite() {
                            sum_y / window as f64
                        } else {
                            y_win.iter().map(|&v| v / window as f64).sum()
                        };
                        y_win
                            .iter()
                            .map(|&v| {
                                let d = v - mean_y;
                                d * d
                            })
                            .sum()
                    }
                } else {
                    y_win.iter().map(|&v| v * v).sum()
                };
                let r_sq = if sst == 0.0 {
                    f64::NAN
                } else {
                    1.0 - sse / sst
                };

                for (c, &b) in beta.iter().enumerate() {
                    coeff_vals[c][idx] = b;
                }
                fitted_vals[idx] = fitted;
                r_sq_vals[idx] = r_sq;
                std_err_vals[idx] = std_err;
            }
        }

        let mut coefficients = Vec::with_capacity(k_cols);
        for vals in coeff_vals {
            coefficients.push(Series::try_new(aligned_dates.clone(), vals)?);
        }
        let r_squared = Series::try_new(aligned_dates.clone(), r_sq_vals)?;
        let fitted_values = Series::try_new(aligned_dates.clone(), fitted_vals)?;
        let standard_deviation_of_errors = Series::try_new(aligned_dates, std_err_vals)?;

        Ok(Self {
            column_indices,
            coefficients,
            r_squared,
            fitted_values,
            standard_deviation_of_errors,
        })
    }

    /// Column indices of the fitted design matrix (`0..k` with intercept, `1..=k` without).
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    /// Rolling coefficient series for column `index`, or `None` if `index` is out of range.
    pub fn coefficient(&self, index: usize) -> Option<&Series> {
        self.column_indices
            .iter()
            .position(|&col| col == index)
            .map(|pos| &self.coefficients[pos])
    }

    /// Rolling coefficient of determination (`R^2`).
    pub fn r_squared(&self) -> &Series {
        &self.r_squared
    }

    /// Rolling fitted values `dot(X[t], beta[t])` at each aligned date.
    pub fn fitted_values(&self) -> &Series {
        &self.fitted_values
    }

    /// Rolling standard deviation of residuals (`sqrt(SSE / (window - k))`).
    pub fn standard_deviation_of_errors(&self) -> &Series {
        &self.standard_deviation_of_errors
    }
}

/// Day-count conventions supported by [`day_count_fraction`].
///
/// Mirrors `gs_quant/datetime/date.py:DayCountConvention` at commit
/// `70220e300524d54bd273be0f4c485d53b3780482`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DayCountConvention {
    /// Actual calendar days divided by `360.0`.
    Actual360,
    /// Actual calendar days divided by `364.0`.
    Actual364,
    /// Actual calendar days divided by `365.25`.
    Actual36525,
    /// Actual calendar days divided by fixed `365.0`.
    Actual365F,
    /// Actual/365L: denominator is `366.0` when `frequency == PaymentFrequency::Annually` and
    /// [`has_feb_29(start, end)`](has_feb_29) is `true`, or when `frequency != Annually` and
    /// `end.year()` is a leap year; otherwise `365.0`.
    Actual365L,
    /// Always returns `1.0`, even when `start >= end`.
    OneOne,
}

/// Payment frequency used by [`day_count_fraction`] (notably [`DayCountConvention::Actual365L`]).
///
/// Discriminants match `gs_quant/datetime/date.py:PaymentFrequency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaymentFrequency {
    Daily = 252,
    Weekly = 52,
    SemiMonthly = 26,
    Monthly = 12,
    SemiQuarterly = 6,
    Quarterly = 4,
    TriAnnually = 3,
    SemiAnnually = 2,
    Annually = 1,
}

/// Business-day rolling rule for [`business_day_offset`] when the input date is not a business day.
///
/// Matches NumPy `np.busday_offset(..., roll=...)` two-step semantics: roll first if needed, then
/// apply `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Roll {
    /// Return `Err(Error::Date)` when `date` is not a business day (maps upstream NumPy
    /// `ValueError`).
    Raise,
    /// Roll to the smallest business day strictly after `date`.
    Forward,
    /// Roll to the largest business day strictly before `date`.
    Backward,
    /// Roll forward; if rolling forward exits `0001-01-01..=9999-12-31` or changes `(year, month)`,
    /// fall back to rolling backward.
    ModifiedFollowing,
    /// Roll backward; if rolling backward exits `0001-01-01..=9999-12-31` or changes
    /// `(year, month)`, fall back to rolling forward.
    ModifiedPreceding,
}

/// Offline business-day calendar defined by an explicit 7-day weekday mask and holiday list.
///
/// Mirrors `numpy.busdaycalendar(weekmask=..., holidays=...)` without any network or dataset
/// dependency.
///
/// # Documented Parity Boundaries
/// - Explicit `HolidayCalendar` construction (`week_mask` + `holidays`) is required; upstream
///   remote `GsCalendar` exchange/currency dataset fetches (`HOLIDAY`, `HOLIDAY_CURRENCY`) are
///   intentionally out of scope for this offline crate.
/// - An all-`false` `week_mask` returns `Err(Error::Date)` (mapping NumPy's `ValueError`).
/// - Valid [`Date`] values span Gregorian `0001-01-01..=9999-12-31`; rolls or offsets that exit
///   this range return `Err(Error::Date)` (whereas `np.datetime64[D]` supports wider 64-bit day
///   offsets outside Python `datetime.date`'s `0001..=9999` range).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HolidayCalendar {
    week_mask: [bool; 7], // Monday = 0 .. Sunday = 6
    holidays: Vec<Date>,  // sorted, deduplicated, filtered to week_mask-valid weekdays
}

impl HolidayCalendar {
    /// Construct a `HolidayCalendar` from a 7-day weekday mask (`Monday = 0` .. `Sunday = 6`) and
    /// a list of holiday dates.
    ///
    /// Returns `Err(Error::Date)` if every entry in `week_mask` is `false`. Holidays falling on
    /// non-business weekdays (`!week_mask[weekday]`) are filtered out, and the remaining holidays
    /// are sorted and deduplicated, matching `numpy.busdaycalendar`.
    pub fn try_new(week_mask: [bool; 7], holidays: Vec<Date>) -> Result<Self, Error> {
        if !week_mask.iter().any(|&b| b) {
            return Err(Error::Date);
        }
        let mut normalized: Vec<Date> = holidays
            .into_iter()
            .filter(|d| week_mask[d.weekday()])
            .collect();
        normalized.sort_unstable();
        normalized.dedup();
        Ok(Self {
            week_mask,
            holidays: normalized,
        })
    }

    /// Construct a standard Monday–Friday calendar (`[true, true, true, true, true, false, false]`)
    /// with the given holidays normalized.
    pub fn weekdays(holidays: Vec<Date>) -> Self {
        Self::try_new([true, true, true, true, true, false, false], holidays)
            .expect("Monday-Friday week mask is non-empty")
    }

    /// Weekday mask (`Monday = 0` .. `Sunday = 6`) where `true` indicates a valid business weekday.
    pub fn week_mask(&self) -> [bool; 7] {
        self.week_mask
    }

    /// Sorted, deduplicated holiday dates falling on valid business weekdays.
    pub fn holidays(&self) -> &[Date] {
        &self.holidays
    }
}

/// Return `true` iff a leap day (`YYYY-02-29`) falls strictly after `start` and on or before `end`
/// (`start < Date(y, 2, 29) <= end`).
///
/// Returns `false` when `start >= end`. Uses the proleptic Gregorian leap-year rule
/// (`y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)`), so `1900` and `2100` are non-leap while
/// `2000` and `2400` are leap.
///
/// # Example
/// ```
/// use quantus_math::{has_feb_29, Date};
///
/// let feb28 = Date::new(2020, 2, 28).unwrap();
/// let feb29 = Date::new(2020, 2, 29).unwrap();
/// let mar01 = Date::new(2020, 3, 1).unwrap();
/// assert!(has_feb_29(feb28, feb29));
/// assert!(!has_feb_29(feb29, mar01));
/// ```
pub fn has_feb_29(start: Date, end: Date) -> bool {
    if start >= end {
        return false;
    }
    for year in start.year..=end.year {
        if is_leap_year(year) {
            let feb29 = Date {
                year,
                month: 2,
                day: 29,
            };
            if start < feb29 && feb29 <= end {
                return true;
            }
        }
    }
    false
}

/// Year fraction between `start` and `end` under the given [`DayCountConvention`] and
/// [`PaymentFrequency`].
///
/// Mirrors `gs_quant/datetime/date.py:day_count_fraction`. [`DayCountConvention::OneOne`] always
/// returns `1.0` even when `start >= end`.
///
/// # Example
/// ```
/// use quantus_math::{day_count_fraction, Date, DayCountConvention, PaymentFrequency};
///
/// let start = Date::new(2020, 3, 1).unwrap();
/// let end = Date::new(2020, 9, 1).unwrap();
/// let annual = day_count_fraction(start, end, DayCountConvention::Actual365L, PaymentFrequency::Annually);
/// let monthly = day_count_fraction(start, end, DayCountConvention::Actual365L, PaymentFrequency::Monthly);
/// assert!((annual - 184.0 / 365.0).abs() < 1e-12);
/// assert!((monthly - 184.0 / 366.0).abs() < 1e-12);
/// ```
pub fn day_count_fraction(
    start: Date,
    end: Date,
    convention: DayCountConvention,
    frequency: PaymentFrequency,
) -> f64 {
    let days = (end.ordinal() - start.ordinal()) as f64;
    match convention {
        DayCountConvention::Actual360 => days / 360.0,
        DayCountConvention::Actual364 => days / 364.0,
        DayCountConvention::Actual36525 => days / 365.25,
        DayCountConvention::Actual365F => days / 365.0,
        DayCountConvention::Actual365L => {
            let denom = if frequency == PaymentFrequency::Annually {
                if has_feb_29(start, end) {
                    366.0
                } else {
                    365.0
                }
            } else if is_leap_year(end.year) {
                366.0
            } else {
                365.0
            };
            days / denom
        }
        DayCountConvention::OneOne => 1.0,
    }
}

/// Return `true` iff `date` is a valid business day under `calendar` (its weekday is enabled in
/// `calendar.week_mask()` and `date` is not in `calendar.holidays()`).
pub fn is_business_day(date: Date, calendar: &HolidayCalendar) -> bool {
    calendar.week_mask[date.weekday()] && calendar.holidays.binary_search(&date).is_err()
}

/// Count week-mask-enabled days in half-open ordinal interval `[start_ord, end_ord)` (`start_ord <= end_ord`).
fn week_mask_days_in_half_open(start_ord: i64, end_ord: i64, week_mask: &[bool; 7]) -> i64 {
    let total_days = end_ord - start_ord;
    if total_days <= 0 {
        return 0;
    }
    let per_week = week_mask.iter().filter(|&&b| b).count() as i64;
    let full_weeks = total_days / 7;
    let rem = (total_days % 7) as usize;
    let start_wd = ((start_ord - 1).rem_euclid(7)) as usize;
    let mut count = full_weeks * per_week;
    for k in 0..rem {
        if week_mask[(start_wd + k) % 7] {
            count += 1;
        }
    }
    count
}

/// Count week-mask-enabled days in `[MIN_ORDINAL, ord]`.
fn week_mask_days_up_to(ord: i64, week_mask: &[bool; 7]) -> i64 {
    if ord < MIN_ORDINAL {
        return 0;
    }
    week_mask_days_in_half_open(MIN_ORDINAL, ord + 1, week_mask)
}

/// Return the ordinal of the 1-based `t`-th week-mask-enabled day starting from `MIN_ORDINAL` (`t >= 1`).
fn nth_week_mask_day(t: i64, week_mask: &[bool; 7]) -> i64 {
    let per_week = week_mask.iter().filter(|&&b| b).count() as i64;
    let idx = t - 1;
    let full_weeks = idx / per_week;
    let mut rem = (idx % per_week) as usize;
    let mut wd = 0usize;
    for (i, &allowed) in week_mask.iter().enumerate() {
        if allowed {
            if rem == 0 {
                wd = i;
                break;
            }
            rem -= 1;
        }
    }
    MIN_ORDINAL + full_weeks * 7 + wd as i64
}

/// Count business days in `[MIN_ORDINAL, ord]` in `O(log H)`.
fn business_days_up_to(ord: i64, calendar: &HolidayCalendar) -> i64 {
    if ord < MIN_ORDINAL {
        return 0;
    }
    let wm = week_mask_days_up_to(ord, &calendar.week_mask);
    let hols = calendar.holidays.partition_point(|h| h.ordinal() <= ord) as i64;
    wm - hols
}

/// Return the 1-based `k`-th business day in `0001-01-01..=9999-12-31` in `O(log H)`.
fn nth_business_day(k: i64, calendar: &HolidayCalendar) -> Result<Date, Error> {
    if k < 1 {
        return Err(Error::Date);
    }
    let mut lo = 0usize;
    let mut hi = calendar.holidays.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let b_before = week_mask_days_up_to(calendar.holidays[mid].ordinal(), &calendar.week_mask)
            - 1
            - mid as i64;
        if b_before < k {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let ord = nth_week_mask_day(k + lo as i64, &calendar.week_mask);
    Date::from_ordinal(ord)
}

/// Shift `date` by `offset` business days under `calendar` using NumPy `np.busday_offset`
/// two-step semantics:
///
/// 1. If `date` is not a business day, roll it according to `roll` (`Roll::Raise` returns
///    `Err(Error::Date)`).
/// 2. Shift the rolled business day by `offset` business days in `O(log H)` time.
///
/// Returns `Err(Error::Date)` if `Roll::Raise` is used on a non-business day or if the rolled or
/// shifted date falls outside Gregorian `0001-01-01..=9999-12-31`.
///
/// # Documented Parity Boundaries
/// - `Roll::Raise` on a non-business day returns `Err(Error::Date)` (mapping NumPy `ValueError`).
/// - Underflow below `0001-01-01` or overflow above `9999-12-31` returns `Err(Error::Date)`
///   (whereas `np.datetime64[D]` supports dates outside Python `datetime.date`'s `0001..=9999`
///   range).
///
/// # Example
/// ```
/// use quantus_math::{business_day_offset, Date, HolidayCalendar, Roll};
///
/// let cal = HolidayCalendar::weekdays(vec![Date::new(2020, 1, 31).unwrap()]);
/// let rolled = business_day_offset(
///     Date::new(2020, 1, 31).unwrap(),
///     0,
///     Roll::ModifiedFollowing,
///     &cal,
/// )
/// .unwrap();
/// assert_eq!(rolled, Date::new(2020, 1, 30).unwrap());
/// ```
pub fn business_day_offset(
    date: Date,
    offset: i32,
    roll: Roll,
    calendar: &HolidayCalendar,
) -> Result<Date, Error> {
    let ord = date.ordinal();
    let max_b = business_days_up_to(MAX_ORDINAL, calendar);
    let (rolled, idx) = if is_business_day(date, calendar) {
        (date, business_days_up_to(ord, calendar))
    } else {
        let r = business_days_up_to(ord, calendar);
        let forward = || -> Result<(Date, i64), Error> {
            let k = r + 1;
            if k > max_b {
                return Err(Error::Date);
            }
            Ok((nth_business_day(k, calendar)?, k))
        };
        let backward = || -> Result<(Date, i64), Error> {
            if r < 1 {
                return Err(Error::Date);
            }
            Ok((nth_business_day(r, calendar)?, r))
        };
        match roll {
            Roll::Raise => return Err(Error::Date),
            Roll::Forward => forward()?,
            Roll::Backward => backward()?,
            Roll::ModifiedFollowing => match forward() {
                Ok((fwd, k)) if fwd.year == date.year && fwd.month == date.month => (fwd, k),
                _ => backward()?,
            },
            Roll::ModifiedPreceding => match backward() {
                Ok((bwd, k)) if bwd.year == date.year && bwd.month == date.month => (bwd, k),
                _ => forward()?,
            },
        }
    };

    if offset == 0 {
        return Ok(rolled);
    }
    let target = idx + i64::from(offset);
    if target < 1 || target > max_b {
        return Err(Error::Date);
    }
    nth_business_day(target, calendar)
}

/// Count business days between `begin` and `end` under `calendar` in `O(log H)` time, matching
/// NumPy `np.busday_count` exact asymmetric interval semantics:
///
/// - When `begin <= end`: `+count(d in [begin, end) where is_business_day(d))` (`begin` inclusive,
///   `end` exclusive).
/// - When `begin > end`: `-count(d in (end, begin] where is_business_day(d))` (`end` exclusive,
///   `begin` inclusive).
///
/// # Example
/// ```
/// use quantus_math::{business_day_count, Date, HolidayCalendar};
///
/// let cal = HolidayCalendar::weekdays(vec![Date::new(2020, 1, 1).unwrap()]);
/// let jan1 = Date::new(2020, 1, 1).unwrap();
/// let jan6 = Date::new(2020, 1, 6).unwrap();
/// assert_eq!(business_day_count(jan1, jan6, &cal), 2);
/// assert_eq!(business_day_count(jan6, jan1, &cal), -3);
/// ```
pub fn business_day_count(begin: Date, end: Date, calendar: &HolidayCalendar) -> i64 {
    if begin == end {
        return 0;
    }
    if begin < end {
        let wm = week_mask_days_in_half_open(begin.ordinal(), end.ordinal(), &calendar.week_mask);
        let lo = calendar.holidays.partition_point(|&h| h < begin);
        let hi = calendar.holidays.partition_point(|&h| h < end);
        wm - (hi - lo) as i64
    } else {
        let wm = week_mask_days_in_half_open(
            end.ordinal() + 1,
            begin.ordinal() + 1,
            &calendar.week_mask,
        );
        let lo = calendar.holidays.partition_point(|&h| h <= end);
        let hi = calendar.holidays.partition_point(|&h| h <= begin);
        -(wm - (hi - lo) as i64)
    }
}

/// Number of calendar days in `(year, month)` under the proleptic Gregorian calendar, or `0` for
/// an invalid month outside `1..=12`.
fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Add `months` calendar months to `start`, clamping the day of month to the target month's
/// length (matching `start + months * dateutil.relativedelta(months=1)` anchored at `start`).
fn add_months_clamped(start: Date, months: i64) -> Result<Date, Error> {
    let total_months = i64::from(start.year - 1) * 12 + i64::from(start.month - 1) + months;
    if total_months < 0 {
        return Err(Error::Date);
    }
    let year_i64 = total_months / 12 + 1;
    if !(1..=9999).contains(&year_i64) {
        return Err(Error::Date);
    }
    let year = year_i64 as i32;
    let month = (total_months % 12 + 1) as u8;
    let day = start.day.min(days_in_month(year, month));
    Date::new(year, month, day)
}

fn intersect_two_sorted(a: &[Date], b: &[Date]) -> Vec<Date> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn align_values(s: &Series, cal: &[Date]) -> Vec<f64> {
    let mut out = Vec::with_capacity(cal.len());
    let mut j = 0;
    for &d in cal {
        while s.dates[j] < d {
            j += 1;
        }
        debug_assert_eq!(s.dates[j], d);
        out.push(s.values[j]);
        j += 1;
    }
    out
}

/// Rebalancing schedule for [`backtest_basket`] and [`basket_series`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalFreq {
    Daily,
    Weekly,
    Monthly,
}

/// Asset weight or execution cost input: either a constant scalar across the aligned calendar
/// or a date-indexed series whose calendar participates in the common date intersection.
#[derive(Debug, Clone, PartialEq)]
pub enum BasketParam<'a> {
    Fixed(f64),
    Series(&'a Series),
}

/// Output of [`backtest_basket`]: basket NAV series (starting at `100.0`) and realized asset
/// weight series in the same asset order as the input price series.
#[derive(Debug, Clone, PartialEq)]
pub struct BasketResult {
    pub nav: Series,
    pub actual_weights: Vec<Series>,
}

/// Offline multi-asset basket backtest returning basket NAV (rebased to `100.0` on the first
/// common date) and realized asset weight series.
///
/// Mirrors upstream `gs_quant/timeseries/backtesting.py:backtest_basket` (lines 52–145) at commit
/// `70220e300524d54bd273be0f4c485d53b3780482`:
///
/// - `weights: None` defaults every asset to fixed equal weight `1.0 / (series.len() as f64)`.
/// - `costs: None` defaults every asset to fixed execution cost `0.0`.
/// - The simulation calendar `cal` is the sorted intersection of `dates()` across all price
///   `series` and every [`BasketParam::Series`] inside `weights` and `costs`.
/// - Rebalance dates:
///   - [`RebalFreq::Daily`]: every date in `cal`.
///   - [`RebalFreq::Weekly`]: hypothetical dates `cal[0] + 7 * k` days for
///     `k in 0..=((cal[n - 1] - cal[0]) / 7)`; each `hyp < cal[n - 1]` snaps to the first
///     `cal[idx] >= hyp` (multiple hypothetical dates snapping to the same `idx` rebalance once;
///     an exact hit `hyp == cal[n - 1]` is excluded, whereas a gap-snapped `hyp < cal[n - 1]`
///     whose next calendar date is `cal[n - 1]` does rebalance on `cal[n - 1]`).
///   - [`RebalFreq::Monthly`]: hypothetical dates `cal[0] + k` calendar months (anchored at
///     `cal[0]` and day-clamped to the target month length, e.g. `2024-01-31 -> 2024-02-29 ->
///     2024-03-31 -> 2024-04-30`), using the same strict `hyp < cal[n - 1]` snap rule.
/// - Recurrence: initializes `nav[0] = 100.0`, `units[a] = 100.0 * w[a][0] / p[a][0]`,
///   `actual_weights[a][0] = w[a][0]`, and `prev_rebal = 0`. For each `i in 1..cal.len()`,
///   accumulates `pnl = sum_a units[a] * (p[a][i] - p[a][i - 1])`, updates
///   `nav_i = nav[i - 1] + pnl`, computes drifted weights
///   `aw[a] = w[a][prev_rebal] * (p[a][i] / p[a][prev_rebal]) * (nav[prev_rebal] / nav_i)`, and
///   on rebalance dates deducts `nav_i -= (sum_a c[a][i] * |w[a][i] - aw[a]|) * nav_i` before
///   resetting `units[a] = nav_i * w[a][i] / p[a][i]`, `aw[a] = w[a][i]`, and `prev_rebal = i`.
///
/// # Documented Parity Boundaries
/// - Only offline `backtest_basket` and [`basket_series`] (excess return) are ported; the remote
///   `Basket` class (`GsAssetApi`/`GsDataApi`) is intentionally out of scope.
/// - `weights: None` and `costs: None` represent upstream defaults (`1/N` equal weight, `0.0`
///   cost); explicit empty slices `Some(&[])` on non-empty `series` return `Err(Error::Length)`
///   (unlike Python's falsy `weights or [...]` / `costs or [...]` fallback).
/// - Empty `series`, mismatched `weights`/`costs` lengths, and an empty/disjoint common calendar
///   intersection all return typed `Err(Error::Length)` (mapping upstream `ZeroDivisionError`,
///   `MqValueError`, and `IndexError`).
/// - Duplicate or unsorted dates and `DatetimeIndex` time zones are rejected by [`Series::try_new`].
///
/// # Example
/// ```
/// use quantus_math::{backtest_basket, BasketParam, Date, RebalFreq, Series};
///
/// let dates = vec![
///     Date::new(2024, 1, 1).unwrap(),
///     Date::new(2024, 1, 2).unwrap(),
///     Date::new(2024, 1, 3).unwrap(),
/// ];
/// let a = Series::try_new(dates.clone(), vec![100.0, 110.0, 121.0]).unwrap();
/// let b = Series::try_new(dates, vec![100.0, 100.0, 100.0]).unwrap();
/// let weights = [BasketParam::Fixed(0.5), BasketParam::Fixed(0.5)];
/// let costs = [BasketParam::Fixed(0.0), BasketParam::Fixed(0.0)];
/// let res = backtest_basket(&[a, b], Some(&weights), Some(&costs), RebalFreq::Daily).unwrap();
/// assert_eq!(res.nav.values()[0], 100.0);
/// assert!((res.nav.values()[1] - 105.0).abs() < 1e-12);
/// assert!((res.nav.values()[2] - 110.25).abs() < 1e-12);
/// assert_eq!(res.actual_weights.len(), 2);
/// ```
pub fn backtest_basket(
    series: &[Series],
    weights: Option<&[BasketParam<'_>]>,
    costs: Option<&[BasketParam<'_>]>,
    rebal_freq: RebalFreq,
) -> Result<BasketResult, Error> {
    let num_assets = series.len();
    if num_assets == 0 {
        return Err(Error::Length);
    }
    if let Some(w) = weights {
        if w.len() != num_assets {
            return Err(Error::Length);
        }
    }
    if let Some(c) = costs {
        if c.len() != num_assets {
            return Err(Error::Length);
        }
    }

    let mut cal = series[0].dates.clone();
    for s in &series[1..] {
        cal = intersect_two_sorted(&cal, &s.dates);
    }
    if let Some(ws) = weights {
        for param in ws {
            if let BasketParam::Series(s) = param {
                cal = intersect_two_sorted(&cal, &s.dates);
            }
        }
    }
    if let Some(cs) = costs {
        for param in cs {
            if let BasketParam::Series(s) = param {
                cal = intersect_two_sorted(&cal, &s.dates);
            }
        }
    }
    if cal.is_empty() {
        return Err(Error::Length);
    }

    let n = cal.len();
    let p: Vec<Vec<f64>> = series.iter().map(|s| align_values(s, &cal)).collect();
    let equal_weight = 1.0 / (num_assets as f64);
    let w: Vec<Vec<f64>> = match weights {
        None => vec![vec![equal_weight; n]; num_assets],
        Some(ws) => ws
            .iter()
            .map(|param| match param {
                BasketParam::Fixed(val) => vec![*val; n],
                BasketParam::Series(s) => align_values(s, &cal),
            })
            .collect(),
    };
    let c: Vec<Vec<f64>> = match costs {
        None => vec![vec![0.0; n]; num_assets],
        Some(cs) => cs
            .iter()
            .map(|param| match param {
                BasketParam::Fixed(val) => vec![*val; n],
                BasketParam::Series(s) => align_values(s, &cal),
            })
            .collect(),
    };

    let rebal_mask = match rebal_freq {
        RebalFreq::Daily => vec![true; n],
        RebalFreq::Weekly => {
            let mut mask = vec![false; n];
            let first = cal[0];
            let last = cal[n - 1];
            let num_rebals = (last.ordinal() - first.ordinal()) / 7;
            for k in 0..=num_rebals {
                let hyp = Date::from_ordinal(first.ordinal() + k * 7)?;
                if hyp < last {
                    let idx = cal.partition_point(|&d| d < hyp);
                    mask[idx] = true;
                }
            }
            mask
        }
        RebalFreq::Monthly => {
            let mut mask = vec![false; n];
            let first = cal[0];
            let last = cal[n - 1];
            let num_rebals = i64::from(last.year() - first.year()) * 12 + i64::from(last.month())
                - i64::from(first.month());
            for k in 0..=num_rebals {
                let hyp = add_months_clamped(first, k)?;
                if hyp < last {
                    let idx = cal.partition_point(|&d| d < hyp);
                    mask[idx] = true;
                }
            }
            mask
        }
    };

    let mut nav = vec![0.0_f64; n];
    let mut units = vec![0.0_f64; num_assets];
    let mut aw = vec![vec![0.0_f64; n]; num_assets];

    nav[0] = 100.0;
    for a in 0..num_assets {
        units[a] = nav[0] * w[a][0] / p[a][0];
        aw[a][0] = w[a][0];
    }
    let mut prev_rebal = 0usize;

    for i in 1..n {
        let mut pnl = 0.0_f64;
        for a in 0..num_assets {
            pnl += units[a] * (p[a][i] - p[a][i - 1]);
        }
        let mut nav_i = nav[i - 1] + pnl;
        for a in 0..num_assets {
            aw[a][i] = w[a][prev_rebal] * (p[a][i] / p[a][prev_rebal]) * (nav[prev_rebal] / nav_i);
        }
        if rebal_mask[i] {
            let mut cost_rate = 0.0_f64;
            for a in 0..num_assets {
                cost_rate += c[a][i] * (w[a][i] - aw[a][i]).abs();
            }
            nav_i -= cost_rate * nav_i;
            for a in 0..num_assets {
                units[a] = nav_i * w[a][i] / p[a][i];
                aw[a][i] = w[a][i];
            }
            prev_rebal = i;
        }
        nav[i] = nav_i;
    }

    let mut actual_weights = Vec::with_capacity(num_assets);
    for asset_weights in aw {
        actual_weights.push(Series::try_new(cal.clone(), asset_weights)?);
    }
    let nav_series = Series::try_new(cal, nav)?;
    Ok(BasketResult {
        nav: nav_series,
        actual_weights,
    })
}

/// Offline multi-asset basket excess-return NAV series (rebased to `100.0` on the first common
/// date).
///
/// Mirrors upstream `gs_quant/timeseries/backtesting.py:basket_series` (lines 148–180) and returns
/// `backtest_basket(series, weights, costs, rebal_freq)?.nav`. See [`backtest_basket`] for full
/// scheduling, cost, alignment, and documented parity-boundary semantics.
///
/// # Example
/// ```
/// use quantus_math::{basket_series, Date, RebalFreq, Series};
///
/// let dates = vec![
///     Date::new(2024, 1, 1).unwrap(),
///     Date::new(2024, 1, 2).unwrap(),
/// ];
/// let a = Series::try_new(dates.clone(), vec![100.0, 120.0]).unwrap();
/// let b = Series::try_new(dates, vec![50.0, 50.0]).unwrap();
/// let nav = basket_series(&[a, b], None, None, RebalFreq::Daily).unwrap();
/// assert_eq!(nav.values(), &[100.0, 110.0]);
/// ```
pub fn basket_series(
    series: &[Series],
    weights: Option<&[BasketParam<'_>]>,
    costs: Option<&[BasketParam<'_>]>,
    rebal_freq: RebalFreq,
) -> Result<Series, Error> {
    Ok(backtest_basket(series, weights, costs, rebal_freq)?.nav)
}

/// One resolved bucketed risk observation matching upstream `gs_quant.risk.core`
/// (`__risk_columns = ("date", "time", "mkt_type", "mkt_asset", "mkt_class", "mkt_point")`
/// plus numeric `value`).
#[derive(Debug, Clone, PartialEq)]
pub struct RiskRow {
    pub date: Option<Date>,
    pub time: Option<String>,
    pub mkt_type: String,
    pub mkt_asset: String,
    pub mkt_class: String,
    pub mkt_point: String,
    pub value: f64,
}

/// A resolved bucketed risk table with an optional unit/currency label (e.g. `Some("USD")`).
#[derive(Debug, Clone, PartialEq)]
pub struct RiskTable {
    pub unit: Option<String>,
    pub rows: Vec<RiskRow>,
}

fn posix_yy(yy: i32) -> i32 {
    if yy <= 68 {
        2000 + yy
    } else {
        1900 + yy
    }
}

fn days_from_ref(target: Date, ref_date: Date) -> f64 {
    (target.ordinal() - ref_date.ordinal()) as f64
}

fn parse_month_title(s: &str) -> Option<u8> {
    match s {
        "Jan" => Some(1),
        "Feb" => Some(2),
        "Mar" => Some(3),
        "Apr" => Some(4),
        "May" => Some(5),
        "Jun" => Some(6),
        "Jul" => Some(7),
        "Aug" => Some(8),
        "Sep" => Some(9),
        "Oct" => Some(10),
        "Nov" => Some(11),
        "Dec" => Some(12),
        _ => None,
    }
}

fn parse_month_upper(s: &str) -> Option<u8> {
    match s {
        "JAN" => Some(1),
        "FEB" => Some(2),
        "MAR" => Some(3),
        "APR" => Some(4),
        "MAY" => Some(5),
        "JUN" => Some(6),
        "JUL" => Some(7),
        "AUG" => Some(8),
        "SEP" => Some(9),
        "OCT" => Some(10),
        "NOV" => Some(11),
        "DEC" => Some(12),
        _ => None,
    }
}

fn parse_month_title_or_upper(s: &str) -> Option<u8> {
    parse_month_title(s).or_else(|| parse_month_upper(s))
}

fn parse_month_ci(s: &str) -> Option<u8> {
    if s.len() != 3 || !s.is_ascii() {
        return None;
    }
    parse_month_upper(&s.to_ascii_uppercase())
}

fn fut_month_code(b: u8) -> Option<u8> {
    match b {
        b'F' => Some(1),
        b'G' => Some(2),
        b'H' => Some(3),
        b'J' => Some(4),
        b'K' => Some(5),
        b'M' => Some(6),
        b'N' => Some(7),
        b'Q' => Some(8),
        b'U' => Some(9),
        b'V' => Some(10),
        b'X' => Some(11),
        b'Z' => Some(12),
        _ => None,
    }
}

fn parse_rdate_part(s: &str) -> Option<(f64, &str)> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut neg_count = 0usize;
    while i < bytes.len() && bytes[i] == b'-' {
        neg_count += 1;
        i += 1;
    }
    if neg_count > 1 {
        return None;
    }
    let digit_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digit_start || i >= bytes.len() {
        return None;
    }
    let scale = match bytes[i] {
        b'd' | b'D' | b'b' | b'B' => 1.0,
        b'w' | b'W' => 7.0,
        b'm' | b'M' | b'f' | b'F' => 30.0,
        b'y' | b'Y' => 365.0,
        _ => return None,
    };
    let num: f64 = s[digit_start..i].parse().ok()?;
    let days = if neg_count == 1 {
        -(num * scale)
    } else {
        num * scale
    };
    Some((days, &s[i + 1..]))
}

fn match_rdate_n(s: &str, min_parts: usize, max_parts: usize) -> Option<f64> {
    let mut rest = s;
    let mut first = 0.0;
    let mut count = 0usize;
    while !rest.is_empty() && count < max_parts {
        let (d, next) = parse_rdate_part(rest)?;
        if count == 0 {
            first = d;
        }
        rest = next;
        count += 1;
    }
    if rest.is_empty() && count >= min_parts {
        Some(first)
    } else {
        None
    }
}

fn parse_two_digits(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() == 2 && b[0].is_ascii_digit() && b[1].is_ascii_digit() {
        Some((b[0] - b'0') * 10 + (b[1] - b'0'))
    } else {
        None
    }
}

fn parse_four_digits(s: &str) -> Option<i32> {
    if s.len() == 4 && s.bytes().all(|b| b.is_ascii_digit()) {
        s.parse::<i32>().ok()
    } else {
        None
    }
}

fn parse_ddmmmyyyy(s: &str) -> Option<Date> {
    if !s.is_ascii() || (s.len() != 8 && s.len() != 9) {
        return None;
    }
    let b = s.as_bytes();
    let day_len = s.len() - 7;
    let day = if day_len == 1 {
        if !b[0].is_ascii_digit() {
            return None;
        }
        b[0] - b'0'
    } else {
        // Matches upstream DDMMMYYYYReg = r"^([1-3]*[0-9]{1}[a-zA-Z]{3}[0-9]{4})$":
        // no leading '0' on two-digit days.
        if !matches!(b[0], b'1'..=b'3') || !b[1].is_ascii_digit() {
            return None;
        }
        (b[0] - b'0') * 10 + (b[1] - b'0')
    };
    let month = parse_month_ci(&s[day_len..day_len + 3])?;
    let year = parse_four_digits(&s[day_len + 3..])?;
    Date::new(year, month, day).ok()
}

fn parse_ddmmmyy(s: &str) -> Option<Date> {
    if !s.is_ascii() || (s.len() != 6 && s.len() != 7) {
        return None;
    }
    let b = s.as_bytes();
    let day_len = s.len() - 5;
    let day = if day_len == 1 {
        if !b[0].is_ascii_digit() {
            return None;
        }
        b[0] - b'0'
    } else {
        // Matches upstream DDMMMYYReg = r"^([0-3]*[0-9]{1}[a-zA-Z]{3}[0-9]{2})$":
        // allows leading '0' (e.g. "05Feb25").
        if !matches!(b[0], b'0'..=b'3') || !b[1].is_ascii_digit() {
            return None;
        }
        (b[0] - b'0') * 10 + (b[1] - b'0')
    };
    let month = parse_month_ci(&s[day_len..day_len + 3])?;
    let yy = i32::from(parse_two_digits(&s[day_len + 3..])?);
    Date::new(posix_yy(yy), month, day).ok()
}

fn parse_bond_coord(s: &str) -> Option<Date> {
    let mut parts = s.split(' ');
    let p0 = parts.next()?;
    let p1 = parts.next()?;
    let p2 = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if !p0.bytes().all(|b| b.is_ascii_digit())
        || !p1.bytes().all(|b| b.is_ascii_digit() || b == b'.')
    {
        return None;
    }
    let b = p2.as_bytes();
    if b.len() != 10 || b[2] != b'/' || b[5] != b'/' {
        return None;
    }
    // Upstream point.py:285 uses format_str = '%d/%m/%Y'.
    let day = parse_two_digits(&p2[0..2])?;
    let month = parse_two_digits(&p2[3..5])?;
    let year = parse_four_digits(&p2[6..10])?;
    Date::new(year, month, day).ok()
}

/// Numeric sort order (approximate days from `ref_date`) for a market data point label,
/// matching upstream `gs_quant.datetime.point.point_sort_order(point, ref_date)`
/// (`gs_quant/datetime/point.py` lines 25–325).
///
/// Returns `None` for patterns where upstream leaves `days = None` (e.g. copula, bond spread,
/// or FX vol addon parameters); [`sort_risk`] treats `None` as `0.0`.
///
/// Unlike upstream Python, Rust requires an explicit `ref_date: Date` instead of reading the
/// system clock via `datetime.date.today()`, ensuring deterministic offline evaluation across
/// dates and hosts (including for the first part of compound `;`-delimited points).
///
/// # Example
/// ```
/// use quantus_math::{point_sort_order, Date};
///
/// let ref_date = Date::new(2024, 1, 15).unwrap();
/// assert_eq!(point_sort_order("O/N", ref_date), Some(0.0));
/// assert_eq!(point_sort_order("3x6", ref_date), Some(90.0));
/// assert_eq!(point_sort_order("1y;5y", ref_date), Some(365.5));
/// assert_eq!(point_sort_order("USD Rho", ref_date), None);
/// ```
pub fn point_sort_order(point: &str, ref_date: Date) -> Option<f64> {
    if point.is_empty() {
        return Some(0.0);
    }

    match point.to_ascii_uppercase().as_str() {
        "O/N" | "DEFAULT" => return Some(0.0),
        "T/N" | "IN" => return Some(0.1),
        "OUT" => return Some(0.2),
        "OIS FIX" => return Some(1.0),
        "CASH STUB" | "CASHSTUB" => return Some(1.1),
        _ => {}
    }

    if point.contains(';') {
        let mut parts = point.split(';').map(str::trim);
        let first_part = parts.next().unwrap_or("");
        let first = match point_sort_order(first_part, ref_date) {
            Some(v) if v != 0.0 => v,
            _ => return Some(0.0),
        };
        let sum_tail: f64 = parts
            .map(|p| point_sort_order(p, ref_date).unwrap_or(0.0))
            .sum();
        return Some(first + (0.1 * sum_tail / first));
    }

    // 1. infl_volReg: (Caplet|ZCCap|Swaption|ZCSwo)
    if let Some((_, val)) = [
        ("Caplet", 0.0),
        ("ZCCap", 1.0),
        ("Swaption", 2.0),
        ("ZCSwo", 3.0),
    ]
    .iter()
    .filter_map(|&(pat, val)| point.find(pat).map(|pos| (pos, val)))
    .min_by_key(|&(pos, _)| pos)
    {
        return Some(val);
    }

    // 2. CopulaReg
    if point.ends_with("Rho")
        || point.ends_with("KO")
        || [
            "Rho Rate", "Rho Vol", "RR", "BF", "Alpha", "Beta", "K0=L", "K0=S",
        ]
        .iter()
        .any(|pat| point.contains(pat))
    {
        return None;
    }

    // 3. SeasonalFrontReg: (Front|Back)
    if let Some((_, val)) = [("Front", 0.0), ("Back", 1.0)]
        .iter()
        .filter_map(|&(pat, val)| point.find(pat).map(|pos| (pos, val)))
        .min_by_key(|&(pos, _)| pos)
    {
        return Some(val);
    }

    // 4. MMMReg: ^(Jan|...|Dec)$
    if let Some(month) = parse_month_title(point) {
        if let Ok(d) = Date::new(2000, month, 1) {
            return Some(days_from_ref(d, ref_date));
        }
    }

    // 5. EuroOrFraReg: ^(Jan|...|Dec|JAN|...|DEC)+([0-9]{2})$
    if point.is_ascii() && point.len() >= 5 && (point.len() - 2).is_multiple_of(3) {
        let prefix = &point[..point.len() - 2];
        if let Some(yy) = parse_two_digits(&point[point.len() - 2..]) {
            let mut last_month = None;
            let mut all_valid = true;
            for chunk in prefix.as_bytes().chunks_exact(3) {
                let s = std::str::from_utf8(chunk).unwrap_or("");
                if let Some(m) = parse_month_title_or_upper(s) {
                    last_month = Some(m);
                } else {
                    all_valid = false;
                    break;
                }
            }
            if all_valid {
                if let Some(m) = last_month {
                    if let Ok(d) = Date::new(posix_yy(i32::from(yy)), m, 15) {
                        return Some(days_from_ref(d, ref_date));
                    }
                }
            }
        }
    }

    // 6. RDatePartReg: 1 or 2 relative date parts
    if let Some(days) = match_rdate_n(point, 1, 2) {
        return Some(days);
    }

    // 7. CashFXReg: 1 or 2 relative date parts followed by " XC"
    if let Some(prefix) = point.strip_suffix(" XC") {
        if let Some(days) = match_rdate_n(prefix, 1, 2) {
            return Some(days);
        }
    }

    // 8. PricerBFReg: 2 or 3 relative date parts
    if let Some(days) = match_rdate_n(point, 2, 3) {
        return Some(days);
    }

    // 9. FRAxReg: ^([0-9]+)x([0-9]+)$
    if let Some((lhs, rhs)) = point.split_once('x') {
        if !lhs.is_empty()
            && !rhs.is_empty()
            && lhs.bytes().all(|b| b.is_ascii_digit())
            && rhs.bytes().all(|b| b.is_ascii_digit())
        {
            if let Ok(num) = lhs.parse::<f64>() {
                return Some(num * 30.0);
            }
        }
    }

    // 10. SpikeQEReg: ^(QE[0-9])-([0-9]{4})$
    if point.is_ascii() && point.len() == 8 && point.starts_with("QE") {
        let b = point.as_bytes();
        if b[2].is_ascii_digit() && b[3] == b'-' {
            if let Some(year) = parse_four_digits(&point[4..8]) {
                let month = match b[2] {
                    b'1' => 3,
                    b'2' => 6,
                    b'3' => 9,
                    _ => 12,
                };
                if let Ok(d) = Date::new(year, month, 1) {
                    return Some(days_from_ref(d, ref_date));
                }
            }
        }
    }

    // 11. MMMYYYYReg: ^([a-zA-Z]{3}[0-9]{4})$
    if point.is_ascii() && point.len() == 7 {
        if let (Some(month), Some(year)) = (
            parse_month_ci(&point[0..3]),
            parse_four_digits(&point[3..7]),
        ) {
            if let Ok(d) = Date::new(year, month, 1) {
                return Some(days_from_ref(d, ref_date));
            }
        }
    }

    // 12. DDMMMYYYYReg
    if let Some(d) = parse_ddmmmyyyy(point) {
        return Some(days_from_ref(d, ref_date));
    }

    // 13. NumberReg: ^([0-9]*)$ (empty already handled above)
    if point.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(v) = point.parse::<f64>() {
            return Some(v);
        }
    }

    // 14. FloatingYear: ^([0-9]*\.[0-9])[yY]$
    if point.is_ascii() && point.len() >= 3 {
        let b = point.as_bytes();
        let n = b.len();
        if matches!(b[n - 1], b'y' | b'Y')
            && b[n - 2].is_ascii_digit()
            && b[n - 3] == b'.'
            && b[..n - 3].iter().all(u8::is_ascii_digit)
        {
            if let Ok(year_val) = point[..n - 1].parse::<f64>() {
                return Some(365.0 * year_val);
            }
        }
    }

    // 15. PricerCoordRegI & PricerCoordRegII
    if let Some(rest) = point
        .strip_prefix("No ")
        .or_else(|| point.strip_prefix("Pricer "))
    {
        if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
            if let Ok(v) = rest.parse::<f64>() {
                return Some(v);
            }
        }
    }

    // 16. PricerBondSpreadReg: ^[0-9][SQHT]([0-9]{2})[/][0-9][SQHT]([0-9]{2})
    if point.is_ascii() && point.len() >= 9 {
        let b = point.as_bytes();
        if b[0].is_ascii_digit()
            && matches!(b[1], b'S' | b'Q' | b'H' | b'T')
            && b[2].is_ascii_digit()
            && b[3].is_ascii_digit()
            && b[4] == b'/'
            && b[5].is_ascii_digit()
            && matches!(b[6], b'S' | b'Q' | b'H' | b'T')
            && b[7].is_ascii_digit()
            && b[8].is_ascii_digit()
        {
            return None;
        }
    }

    // 17. LYYReg: ([FGHJKMNQUVXZ])([0-9]{2})
    {
        let b = point.as_bytes();
        for i in 0..b.len().saturating_sub(2) {
            if let Some(month) = fut_month_code(b[i]) {
                if b[i + 1].is_ascii_digit() && b[i + 2].is_ascii_digit() {
                    let yy = i32::from((b[i + 1] - b'0') * 10 + (b[i + 2] - b'0'));
                    if let Ok(d) = Date::new(posix_yy(yy), month, 1) {
                        return Some(days_from_ref(d, ref_date));
                    }
                }
            }
        }
    }

    // 18. DatePairReg (YYYYMMDD/YYYYMMDD) & DatePairReg2 (YYYYMMDD YYYYMMDD)
    if point.is_ascii() && point.len() == 17 {
        let b = point.as_bytes();
        if matches!(b[8], b'/' | b' ')
            && b[0..8].iter().all(u8::is_ascii_digit)
            && b[9..17].iter().all(u8::is_ascii_digit)
        {
            if let (Some(year), Some(month), Some(day)) = (
                parse_four_digits(&point[9..13]),
                parse_two_digits(&point[13..15]),
                parse_two_digits(&point[15..17]),
            ) {
                if let Ok(d) = Date::new(year, month, day) {
                    return Some(days_from_ref(d, ref_date));
                }
            }
        }
    }

    // 19. MMMYYReg: ^(JAN|...|DEC) ([0-9]{2})
    if point.is_ascii() && point.len() >= 6 && point.as_bytes()[3] == b' ' {
        if let (Some(month), Some(yy)) = (
            parse_month_upper(&point[0..3]),
            parse_two_digits(&point[4..6]),
        ) {
            if let Ok(d) = Date::new(posix_yy(i32::from(yy)), month, 1) {
                return Some(days_from_ref(d, ref_date));
            }
        }
    }

    // 20. FXVolAddonParmsReg
    if [
        "Spread Addon",
        "Spread Init",
        "Spread Final",
        "Rho Addon",
        "Rho Init",
        "Rho Final",
        "Vol Addon",
        "Vol Init",
        "Vol Final",
        "HL",
        "Addon HL",
    ]
    .iter()
    .any(|pat| point.contains(pat))
    {
        return None;
    }

    // 21. BondCoordReg: ^[0-9]* ([0-9.]*) ([0-9]{2}/[0-9]{2}/[0-9]{4})$
    if let Some(d) = parse_bond_coord(point) {
        return Some(days_from_ref(d, ref_date));
    }

    // 22. BondFutReg (^[A-Z]{3}([FGHJKMNQUVXZ])([0-9])$) & FFFutReg (^FF([FGHJKMNQUVXZ])([0-9])$)
    if point.is_ascii() {
        let b = point.as_bytes();
        if b.len() == 5 && b[0..3].iter().all(u8::is_ascii_uppercase) && b[4].is_ascii_digit() {
            if let Some(month) = fut_month_code(b[3]) {
                if let Ok(d) = Date::new(ref_date.year(), month, 1) {
                    return Some(days_from_ref(d, ref_date));
                }
            }
        }
        if b.len() == 4 && &b[0..2] == b"FF" && b[3].is_ascii_digit() {
            if let Some(month) = fut_month_code(b[2]) {
                if let Ok(d) = Date::new(ref_date.year(), month, 1) {
                    return Some(days_from_ref(d, ref_date));
                }
            }
        }
    }

    // 23. RepoGCReg: ^(ON|SN|TN|[0-9]+) (|Month |Week |Year |Day )GC$
    match point {
        "ON GC" => return Some(0.0),
        "TN GC" => return Some(1.0),
        "SN GC" => return Some(2.0),
        _ => {}
    }
    if let Some(rest) = point.strip_suffix(" GC") {
        if let Some((num_s, unit_s)) = rest.split_once(' ') {
            if !num_s.is_empty() && num_s.bytes().all(|b| b.is_ascii_digit()) {
                let scale = match unit_s {
                    "Day" => Some(1.0),
                    "Week" => Some(7.0),
                    "Month" => Some(30.0),
                    "Year" => Some(365.0),
                    _ => None,
                };
                if let (Some(scale), Ok(num)) = (scale, num_s.parse::<f64>()) {
                    return Some(num * scale);
                }
            }
        } else if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
            // Upstream RepoGCReg matches e.g. "3 GC" with empty unit -> leaves days = None.
            return None;
        }
    }

    // 24. RelativeReg: ^([0-9]+) (day|week|month|year|DAY|WEEK|MONTH|YEAR)$
    if let Some((num_s, unit_s)) = point.split_once(' ') {
        if !num_s.is_empty() && num_s.bytes().all(|b| b.is_ascii_digit()) {
            let scale = match unit_s {
                "day" | "DAY" => Some(1.0),
                "week" | "WEEK" => Some(7.0),
                "month" | "MONTH" => Some(30.0),
                "year" | "YEAR" => Some(365.0),
                _ => None,
            };
            if let (Some(scale), Ok(num)) = (scale, num_s.parse::<f64>()) {
                return Some(num * scale);
            }
        }
    }

    // 25. DDMMMYYReg
    if let Some(d) = parse_ddmmmyy(point) {
        return Some(days_from_ref(d, ref_date));
    }

    // 26. Fallback
    Some(0.0)
}

fn normalized_point_order(point: &str, ref_date: Date) -> f64 {
    let v = point_sort_order(point, ref_date).unwrap_or(0.0);
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// Stable-sort bucketed risk rows in upstream `__risk_columns` order:
/// `(date, time, mkt_type, mkt_asset, mkt_class, point_sort_order(mkt_point, ref_date).unwrap_or(0.0))`.
///
/// Mirrors upstream `gs_quant/risk/core.py:sort_risk` (lines 651–682). When all six sort keys
/// tie (for example `"30d"` and `"1m"`, both `30.0` days), Rust's stable sort preserves the
/// input row order, matching Python's Timsort.
///
/// # Example
/// ```
/// use quantus_math::{sort_risk, Date, RiskRow, RiskTable};
///
/// let ref_date = Date::new(2024, 1, 15).unwrap();
/// let table = RiskTable {
///     unit: Some("USD".into()),
///     rows: vec![
///         RiskRow {
///             date: None, time: None, mkt_type: "IR".into(), mkt_asset: "USD".into(),
///             mkt_class: "SWAP".into(), mkt_point: "5y".into(), value: 10.0,
///         },
///         RiskRow {
///             date: None, time: None, mkt_type: "IR".into(), mkt_asset: "USD".into(),
///             mkt_class: "SWAP".into(), mkt_point: "1y".into(), value: 20.0,
///         },
///     ],
/// };
/// let sorted = sort_risk(&table, ref_date);
/// assert_eq!(sorted.rows[0].mkt_point, "1y");
/// assert_eq!(sorted.rows[1].mkt_point, "5y");
/// ```
pub fn sort_risk(table: &RiskTable, ref_date: Date) -> RiskTable {
    let mut rows = table.rows.clone();
    rows.sort_by(|a, b| {
        a.date
            .cmp(&b.date)
            .then_with(|| a.time.cmp(&b.time))
            .then_with(|| a.mkt_type.cmp(&b.mkt_type))
            .then_with(|| a.mkt_asset.cmp(&b.mkt_asset))
            .then_with(|| a.mkt_class.cmp(&b.mkt_class))
            .then_with(|| {
                normalized_point_order(&a.mkt_point, ref_date)
                    .total_cmp(&normalized_point_order(&b.mkt_point, ref_date))
            })
    });
    RiskTable {
        unit: table.unit.clone(),
        rows,
    }
}

/// Aggregate one or more resolved bucketed risk tables.
///
/// Mirrors upstream `gs_quant/risk/core.py:aggregate_risk` and `aggregate_results` (lines 516–618):
///
/// - Empty `tables` slice returns `Err(Error::Length)`.
/// - Validates unit compatibility across all tables with non-empty `unit` labels, returning
///   `Err(Error::UnitMismatch)` when two non-empty units differ and inheriting a non-empty unit
///   when other tables have `None` or `Some("")`.
/// - Replaces `NaN` values with `0.0` (`fillna(0)`) while propagating `±Inf`.
/// - Groups and sums by `(date, time, mkt_type, mkt_asset, mkt_class, mkt_point)` in canonical
///   lexicographical order (matching pandas `groupby(..., sort=True)` before `sort_risk`).
/// - Optionally filters buckets by strict `value.abs() > threshold` (so `== threshold`, `0.0` at
///   `threshold = 0.0`, and all buckets at `threshold = NaN` or `+Inf` are dropped).
/// - Returns the final [`RiskTable`] sorted via [`sort_risk`].
///
/// Documented parity boundaries: only already-resolved bucketed risk tables/rows are supported
/// (remote `PricingContext`, `Future`, server pricing, and `ErrorValue` are out of scope), and
/// `ref_date: Date` is passed explicitly instead of reading `datetime.date.today()`.
///
/// # Example
/// ```
/// use quantus_math::{aggregate_risk, Date, RiskRow, RiskTable};
///
/// let ref_date = Date::new(2024, 1, 15).unwrap();
/// let row = |pt: &str, v: f64| RiskRow {
///     date: None, time: None, mkt_type: "IR".into(), mkt_asset: "USD".into(),
///     mkt_class: "SWAP".into(), mkt_point: pt.into(), value: v,
/// };
/// let t1 = RiskTable { unit: Some("USD".into()), rows: vec![row("5y", 10.0), row("1y", 20.0)] };
/// let t2 = RiskTable { unit: None, rows: vec![row("5y", -10.0), row("1y", f64::NAN)] };
/// let agg = aggregate_risk(&[t1, t2], Some(0.0), ref_date).unwrap();
/// assert_eq!(agg.unit.as_deref(), Some("USD"));
/// assert_eq!(agg.rows.len(), 1);
/// assert_eq!(agg.rows[0].mkt_point, "1y");
/// assert_eq!(agg.rows[0].value, 20.0);
/// ```
pub fn aggregate_risk(
    tables: &[RiskTable],
    threshold: Option<f64>,
    ref_date: Date,
) -> Result<RiskTable, Error> {
    if tables.is_empty() {
        return Err(Error::Length);
    }

    let mut unit: Option<String> = None;
    for table in tables {
        if let Some(u) = &table.unit {
            if !u.is_empty() {
                match &unit {
                    Some(existing) if existing != u => return Err(Error::UnitMismatch),
                    None => unit = Some(u.clone()),
                    _ => {}
                }
            }
        }
    }

    type BucketKey = (Option<Date>, Option<String>, String, String, String, String);
    let mut buckets: BTreeMap<BucketKey, f64> = BTreeMap::new();
    for table in tables {
        for row in &table.rows {
            let val = if row.value.is_nan() { 0.0 } else { row.value };
            let key = (
                row.date,
                row.time.clone(),
                row.mkt_type.clone(),
                row.mkt_asset.clone(),
                row.mkt_class.clone(),
                row.mkt_point.clone(),
            );
            *buckets.entry(key).or_insert(0.0) += val;
        }
    }

    let mut rows = Vec::with_capacity(buckets.len());
    for ((date, time, mkt_type, mkt_asset, mkt_class, mkt_point), value) in buckets {
        let keep = match threshold {
            Some(th) => value.abs() > th,
            None => true,
        };
        if keep {
            rows.push(RiskRow {
                date,
                time,
                mkt_type,
                mkt_asset,
                mkt_class,
                mkt_point,
                value,
            });
        }
    }

    Ok(sort_risk(&RiskTable { unit, rows }, ref_date))
}

/// Subtract bucketed risk (`left - right`).
///
/// Validates unit compatibility (`Err(Error::UnitMismatch)`) and optional `date`/`time` presence
/// compatibility when both tables are non-empty (`Err(Error::Length)`), negates `right`'s values,
/// and delegates to [`aggregate_risk`].
///
/// Documented parity boundary: upstream `gs_quant/risk/core.py:642-643` asserts
/// `'value' in left.columns.names`, which inspects pandas `Index.names` (`[None]`) instead of
/// column labels and raises `AssertionError` on standard DataFrames. Rust implements the
/// documented/intended semantics (`aggregate_risk((left, -right))` with unit and dimension
/// checks).
///
/// # Example
/// ```
/// use quantus_math::{subtract_risk, Date, RiskRow, RiskTable};
///
/// let ref_date = Date::new(2024, 1, 15).unwrap();
/// let row = |pt: &str, v: f64| RiskRow {
///     date: None, time: None, mkt_type: "IR".into(), mkt_asset: "USD".into(),
///     mkt_class: "SWAP".into(), mkt_point: pt.into(), value: v,
/// };
/// let left = RiskTable { unit: Some("USD".into()), rows: vec![row("5y", 10.0), row("1y", 25.0)] };
/// let right = RiskTable { unit: Some("USD".into()), rows: vec![row("5y", 10.0), row("2y", 4.0)] };
/// let diff = subtract_risk(&left, &right, ref_date).unwrap();
/// assert_eq!(diff.rows.len(), 3);
/// assert_eq!(diff.rows[0].value, 25.0);
/// assert_eq!(diff.rows[1].value, -4.0);
/// assert_eq!(diff.rows[2].value, 0.0);
/// ```
pub fn subtract_risk(
    left: &RiskTable,
    right: &RiskTable,
    ref_date: Date,
) -> Result<RiskTable, Error> {
    if let (Some(lu), Some(ru)) = (&left.unit, &right.unit) {
        if !lu.is_empty() && !ru.is_empty() && lu != ru {
            return Err(Error::UnitMismatch);
        }
    }

    if !left.rows.is_empty() && !right.rows.is_empty() {
        let left_has_date = left.rows.iter().any(|r| r.date.is_some());
        let right_has_date = right.rows.iter().any(|r| r.date.is_some());
        let left_has_time = left.rows.iter().any(|r| r.time.is_some());
        let right_has_time = right.rows.iter().any(|r| r.time.is_some());
        if left_has_date != right_has_date || left_has_time != right_has_time {
            return Err(Error::Length);
        }
    }

    let right_negated = RiskTable {
        unit: right.unit.clone(),
        rows: right
            .rows
            .iter()
            .map(|r| RiskRow {
                date: r.date,
                time: r.time.clone(),
                mkt_type: r.mkt_type.clone(),
                mkt_asset: r.mkt_asset.clone(),
                mkt_class: r.mkt_class.clone(),
                mkt_point: r.mkt_point.clone(),
                value: -r.value,
            })
            .collect(),
    };

    aggregate_risk(&[left.clone(), right_negated], None, ref_date)
}
