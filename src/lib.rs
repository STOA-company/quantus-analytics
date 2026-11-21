//! Offline date-indexed prices, moving averages, and returns.
//! Adapted from gs_quant/timeseries/{technicals,econometrics,helper}.py at
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
