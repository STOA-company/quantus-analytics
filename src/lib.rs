//! Offline date-indexed prices, moving averages, returns, realized volatility, correlation, and beta.
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
