//! Offline date-indexed prices, price level reconstruction from returns, moving averages, exponential moving averages, smoothed moving averages, relative strength index, moving average convergence divergence (MACD), rolling standard deviation (`std`), returns, realized volatility, correlation, beta, and maximum drawdown.
//! Adapted from gs_quant/timeseries/{technicals,econometrics,statistics,datetime,helper}.py at
//! 70220e300524d54bd273be0f4c485d53b3780482; Rust translation and modifications by STOA.
//! Copyright 2018-2019 Goldman Sachs. Upstream also includes code copyright Scott Weinstein.
//! Licensed under Apache-2.0; see LICENSE and both NOTICE files.

use std::fmt;

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
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
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
    fn ordinal(self) -> i64 {
        let y = i64::from(self.year - 1);
        let mut n = 365 * y + y / 4 - y / 100 + y / 400 + i64::from(self.day);
        for m in 1..self.month {
            n += match m {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                _ if self.year % 4 == 0 && (self.year % 100 != 0 || self.year % 400 == 0) => 29,
                _ => 28,
            };
        }
        n
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
