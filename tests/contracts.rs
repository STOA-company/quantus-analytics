use quantus_math::{
    aggregate_risk, backtest_basket, basket_series, beta, bollinger_bands, business_day_count,
    business_day_offset, corr_swap_correlation, correlation, cov, day_count_fraction,
    exponential_moving_average, has_feb_29, is_business_day, macd, max_drawdown, moving_average,
    point_sort_order, prices, relative_strength_index, returns, smoothed_moving_average, sort_risk,
    std as series_std, subtract_risk, vol_swap_volatility, volatility, BasketParam, BollingerBands,
    Date, DayCountConvention, Error, HolidayCalendar, PaymentFrequency, RebalFreq, ReturnKind,
    RiskRow, RiskTable, Roll, RollingLinearRegression, Series, Window,
};
fn date(y: i32, m: u8, d: u8) -> Date {
    Date::new(y, m, d).unwrap()
}
#[test]
fn validates_dates_and_series() {
    assert!(Date::new(2024, 2, 29).is_ok());
    assert_eq!(Date::new(1900, 2, 29), Err(Error::Date));
    assert_eq!(Date::new(0, 1, 1), Err(Error::Date));
    assert_eq!(
        Series::try_new(vec![date(2024, 1, 1)], vec![]),
        Err(Error::Length)
    );
    assert_eq!(
        Series::try_new(vec![date(2024, 1, 1); 2], vec![1., 2.]),
        Err(Error::UnsortedOrDuplicate)
    );
}
#[test]
fn invalid_windows_empty_and_lag() {
    let s = Series::try_new(vec![date(2024, 1, 1)], vec![1.]).unwrap();
    assert_eq!(
        moving_average(&s, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        moving_average(&s, Window::Observations { size: 1, ramp: 2 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        moving_average(&s, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(returns(&s, 0, ReturnKind::Simple), Err(Error::Lag));
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(moving_average(&empty, Window::Days { size: 0, ramp: 0 })
        .unwrap()
        .values()
        .is_empty());
    assert!(returns(&empty, 1, ReturnKind::Simple)
        .unwrap()
        .values()
        .is_empty());
}
#[test]
fn calendar_window_across_leap_day() {
    let s = Series::try_new(
        vec![
            date(2024, 2, 28),
            date(2024, 2, 29),
            date(2024, 3, 1),
            date(2024, 3, 3),
        ],
        vec![1., 2., 3., 4.],
    )
    .unwrap();
    let result = moving_average(&s, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(result.dates(), &[date(2024, 3, 1), date(2024, 3, 3)]);
    assert_eq!(result.values(), &[2.5, 4.]);
}

#[test]
fn volatility_prices_returns_windows_and_zero_mean() {
    let prices = Series::try_new(
        (1..=6).map(|d| date(2019, 1, d)).collect(),
        vec![100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792],
    )
    .unwrap();
    let v = volatility(
        &prices,
        Window::Expanding,
        Some(ReturnKind::Simple),
        None,
        false,
    )
    .unwrap();
    assert_eq!(v.dates().len(), 6);
    assert!(v.values()[0].is_nan() && v.values()[1].is_nan());
    assert!((v.values()[2] - 11.224972160321835).abs() < 1e-10);
    assert!((v.values()[5] - 26.563132345414413).abs() < 1e-10);

    let rets = Series::try_new(
        vec![
            date(2024, 2, 26),
            date(2024, 2, 27),
            date(2024, 2, 29),
            date(2024, 3, 1),
        ],
        vec![0.01, 0.03, f64::NAN, 0.05],
    )
    .unwrap();
    let sample_days = volatility(
        &rets,
        Window::Days { size: 3, ramp: 1 },
        None,
        Some(252),
        false,
    )
    .unwrap();
    assert_eq!(
        sample_days.dates(),
        &[date(2024, 2, 27), date(2024, 2, 29), date(2024, 3, 1)]
    );
    assert!((sample_days.values()[0] - 22.44994432064365).abs() < 1e-10);
    assert!(sample_days.values()[1].is_nan() && sample_days.values()[2].is_nan());

    let rms_days = volatility(
        &rets,
        Window::Days { size: 3, ramp: 1 },
        None,
        Some(252),
        true,
    )
    .unwrap();
    assert!((rms_days.values()[0] - 35.4964786985977).abs() < 1e-10);
    assert!((rms_days.values()[1] - 47.62352359916263).abs() < 1e-10);
    assert!((rms_days.values()[2] - 79.37253933193773).abs() < 1e-10);

    let zero_af = volatility(
        &rets,
        Window::Days { size: 3, ramp: 0 },
        None,
        Some(0),
        false,
    )
    .unwrap();
    assert!(zero_af.values()[0].is_nan());
    assert_eq!(zero_af.values()[1], 0.0);
    assert!(zero_af.values()[2].is_nan() && zero_af.values()[3].is_nan());

    let abs_vol = volatility(
        &prices,
        Window::Observations { size: 3, ramp: 0 },
        Some(ReturnKind::Absolute),
        Some(1),
        false,
    )
    .unwrap();
    assert!(abs_vol.values()[0].is_nan() && abs_vol.values()[1].is_nan());
    assert!(abs_vol.values()[2].is_finite());
}

#[test]
fn volatility_errors_empty_oversize_and_auto_bands() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(volatility(
        &empty,
        Window::Observations { size: 0, ramp: 0 },
        Some(ReturnKind::Simple),
        None,
        false
    )
    .unwrap()
    .values()
    .is_empty());

    let single = Series::try_new(vec![date(2024, 2, 26)], vec![0.01]).unwrap();
    assert_eq!(
        volatility(&single, Window::Expanding, None, None, false),
        Err(Error::AnnualizationFactor)
    );
    assert!(
        volatility(&single, Window::Expanding, None, Some(252), false)
            .unwrap()
            .values()[0]
            .is_nan()
    );
    assert!(
        (volatility(&single, Window::Expanding, None, Some(252), true)
            .unwrap()
            .values()[0]
            - 15.874507866387544)
            .abs()
            < 1e-10
    );

    let pair =
        Series::try_new(vec![date(2024, 2, 26), date(2024, 2, 27)], vec![0.01, 0.03]).unwrap();
    assert_eq!(
        volatility(
            &pair,
            Window::Observations { size: 0, ramp: 0 },
            None,
            Some(252),
            false
        ),
        Err(Error::Window)
    );
    assert_eq!(
        volatility(
            &pair,
            Window::Days { size: 0, ramp: 0 },
            None,
            Some(252),
            false
        ),
        Err(Error::Window)
    );
    assert_eq!(
        volatility(
            &pair,
            Window::Observations { size: 2, ramp: 3 },
            None,
            Some(252),
            false
        ),
        Err(Error::Ramp)
    );
    assert!(volatility(
        &pair,
        Window::Observations { size: 2, ramp: 2 },
        None,
        None,
        false
    )
    .unwrap()
    .values()
    .is_empty());
    assert!(volatility(
        &pair,
        Window::Observations { size: 3, ramp: 0 },
        None,
        Some(252),
        false
    )
    .unwrap()
    .values()
    .is_empty());
    assert_eq!(
        volatility(
            &pair,
            Window::Observations { size: 3, ramp: 0 },
            None,
            None,
            false
        ),
        Err(Error::AnnualizationFactor)
    );

    let non_pos =
        Series::try_new(vec![date(2024, 2, 26), date(2024, 2, 27)], vec![100.0, 0.0]).unwrap();
    assert_eq!(
        volatility(
            &non_pos,
            Window::Observations { size: 5, ramp: 0 },
            Some(ReturnKind::Logarithmic),
            Some(252),
            false
        ),
        Err(Error::NonPositivePrice)
    );

    // Auto annualization: accepted bands and rejected gaps
    let irregular = Series::try_new(
        vec![date(2024, 2, 26), date(2024, 2, 29), date(2024, 3, 3)],
        vec![0.01, 0.03, 0.02],
    )
    .unwrap();
    assert_eq!(
        volatility(&irregular, Window::Expanding, None, None, false),
        Err(Error::AnnualizationFactor)
    );

    // Large offset two-pass stability: stays close to true spread without cancellation
    let shifted = Series::try_new(
        vec![
            date(2024, 2, 26),
            date(2024, 2, 27),
            date(2024, 2, 28),
            date(2024, 2, 29),
        ],
        vec![1e9 + 0.1, 1e9 + 0.2, 1e9 + 0.3, 1e9 + 0.5],
    )
    .unwrap();
    let shifted_vol = volatility(
        &shifted,
        Window::Observations { size: 3, ramp: 0 },
        None,
        Some(252),
        false,
    )
    .unwrap();
    assert!((shifted_vol.values()[1] - 112.24972160321824).abs() < 1e-3);
    assert!((shifted_vol.values()[3] - 242.4871130596428).abs() < 1e-3);
}

#[test]
fn correlation_prices_windows_mixed_returns_and_determinism() {
    let x = Series::try_new(
        vec![
            date(2024, 3, 1),
            date(2024, 3, 2),
            date(2024, 3, 3),
            date(2024, 3, 5),
            date(2024, 3, 6),
        ],
        vec![10.0, 12.0, 14.0, 13.0, 16.0],
    )
    .unwrap();
    let y = Series::try_new(
        vec![
            date(2024, 3, 1),
            date(2024, 3, 3),
            date(2024, 3, 4),
            date(2024, 3, 5),
            date(2024, 3, 6),
        ],
        vec![20.0, 24.0, 22.0, 19.0, 25.0],
    )
    .unwrap();

    // Expanding default on mismatched calendars, re-run determinism
    let run1 = correlation(
        &x,
        &y,
        Window::Expanding,
        Some((ReturnKind::Simple, ReturnKind::Simple)),
        false,
    )
    .unwrap();
    let run2 = correlation(
        &x,
        &y,
        Window::Expanding,
        Some((ReturnKind::Simple, ReturnKind::Simple)),
        false,
    )
    .unwrap();
    assert_eq!(run1.dates(), run2.dates());
    for (&a, &b) in run1.values().iter().zip(run2.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(run1.dates(), x.dates());
    assert!(run1.values()[0].is_nan() && run1.values()[1].is_nan() && run1.values()[2].is_nan());
    assert!((run1.values()[3] - 1.0).abs() < 1e-12);
    assert!((run1.values()[4] - 0.9989243898423678).abs() < 1e-10);

    // Mixed return kinds (Logarithmic, Absolute): observation window, observation ramp, calendar ramp
    let mixed = Some((ReturnKind::Logarithmic, ReturnKind::Absolute));
    let obs = correlation(
        &x,
        &y,
        Window::Observations { size: 3, ramp: 0 },
        mixed,
        false,
    )
    .unwrap();
    assert_eq!(obs.dates(), x.dates());
    assert!(obs.values()[0].is_nan() && obs.values()[1].is_nan() && obs.values()[2].is_nan());
    assert!((obs.values()[3] - 1.0).abs() < 1e-12);
    assert!((obs.values()[4] - 0.9994382571107351).abs() < 1e-10);

    let obs_ramp = correlation(
        &x,
        &y,
        Window::Observations { size: 3, ramp: 2 },
        mixed,
        false,
    )
    .unwrap();
    assert_eq!(
        obs_ramp.dates(),
        &[date(2024, 3, 3), date(2024, 3, 5), date(2024, 3, 6)]
    );
    assert!(obs_ramp.values()[0].is_nan());
    assert!((obs_ramp.values()[1] - 1.0).abs() < 1e-12);
    assert!((obs_ramp.values()[2] - 0.9994382571107351).abs() < 1e-10);

    let cal_ramp = correlation(&x, &y, Window::Days { size: 3, ramp: 2 }, mixed, false).unwrap();
    assert_eq!(
        cal_ramp.dates(),
        &[date(2024, 3, 3), date(2024, 3, 5), date(2024, 3, 6)]
    );
    assert!(cal_ramp.values()[0].is_nan());
    assert!((cal_ramp.values()[1] - 1.0).abs() < 1e-12);
    assert!((cal_ramp.values()[2] - 1.0).abs() < 1e-12);

    // Calendar window across leap day (2024-02-29)
    let leap_x = Series::try_new(
        vec![
            date(2024, 2, 26),
            date(2024, 2, 28),
            date(2024, 2, 29),
            date(2024, 3, 1),
            date(2024, 3, 4),
            date(2024, 3, 6),
        ],
        vec![0.01, 0.03, f64::NAN, 0.02, 0.01, 0.04],
    )
    .unwrap();
    let leap_y = Series::try_new(
        vec![
            date(2024, 2, 26),
            date(2024, 2, 27),
            date(2024, 2, 28),
            date(2024, 2, 29),
            date(2024, 3, 4),
            date(2024, 3, 6),
        ],
        vec![0.02, 0.03, -0.02, 0.04, 0.03, 0.01],
    )
    .unwrap();
    let leap_3d = correlation(
        &leap_x,
        &leap_y,
        Window::Days { size: 3, ramp: 0 },
        None,
        false,
    )
    .unwrap();
    assert!(leap_3d.values()[0].is_nan());
    assert!((leap_3d.values()[1] - (-1.0)).abs() < 1e-12);
    assert!(leap_3d.values()[2].is_nan());
    assert!(leap_3d.values()[3].is_nan());
    assert!(leap_3d.values()[4].is_nan());
    assert!((leap_3d.values()[5] - (-1.0)).abs() < 1e-12);
}

#[test]
fn correlation_zero_variance_zero_mean_ramps_and_errors() {
    let gap = Series::try_new(
        vec![
            date(2024, 3, 1),
            date(2024, 3, 2),
            date(2024, 3, 3),
            date(2024, 3, 5),
            date(2024, 3, 6),
        ],
        vec![0.01, f64::NAN, 0.02, 0.04, 0.01],
    )
    .unwrap();
    let const_001 = Series::try_new(gap.dates().to_vec(), vec![0.01; 5]).unwrap();
    let const_zero = Series::try_new(gap.dates().to_vec(), vec![0.0; 5]).unwrap();

    // Zero variance yields NaN for both 0.0 and non-power-of-two constants (0.01)
    for c in [&const_001, &const_zero] {
        for w in [
            Window::Observations { size: 3, ramp: 0 },
            Window::Days { size: 3, ramp: 0 },
        ] {
            let res = correlation(&gap, c, w, None, false).unwrap();
            assert_eq!(res.dates(), gap.dates());
            assert!(res.values().iter().all(|v| v.is_nan()));
        }
    }

    // Zero variance on 0.1 where 0.1 + 0.1 + 0.1 != 0.3: both-constant and one-side-constant
    // must yield NaN across observation, calendar-day, and expanding windows.
    let const_dates = vec![date(2024, 3, 1), date(2024, 3, 2), date(2024, 3, 3)];
    let const_01 = Series::try_new(const_dates.clone(), vec![0.1; 3]).unwrap();
    let varying_01 = Series::try_new(const_dates.clone(), vec![1.0, 2.0, 3.0]).unwrap();
    for w in [
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
        Window::Expanding,
    ] {
        for (left, right) in [
            (&const_01, &const_01),
            (&const_01, &varying_01),
            (&varying_01, &const_01),
        ] {
            let res = correlation(left, right, w, None, false).unwrap();
            assert_eq!(res.dates(), const_dates.as_slice());
            assert!(res.values().iter().all(|v| v.is_nan()));
        }
    }
    // Zero-mean mode does not center; identical non-zero constant series keeps correlation 1.0
    let zm_const_01 = correlation(
        &const_01,
        &const_01,
        Window::Observations { size: 3, ramp: 0 },
        None,
        true,
    )
    .unwrap();
    assert!((zm_const_01.values()[2] - 1.0).abs() < 1e-12);

    // Zero-mean mode with dense matching returns and expanding / observation windows
    let dates = vec![
        date(2024, 2, 26),
        date(2024, 2, 28),
        date(2024, 2, 29),
        date(2024, 3, 1),
        date(2024, 3, 4),
        date(2024, 3, 6),
    ];
    let xd = Series::try_new(dates.clone(), vec![0.01, 0.03, 0.02, 0.02, 0.01, 0.04]).unwrap();
    let yd = Series::try_new(dates.clone(), vec![0.02, -0.02, 0.04, 0.01, 0.03, 0.01]).unwrap();
    let zm_exp = correlation(&xd, &yd, Window::Expanding, None, true).unwrap();
    assert_eq!(zm_exp.dates(), xd.dates());
    assert!((zm_exp.values()[0] - 1.0).abs() < 1e-12);
    assert!(zm_exp.values()[5].is_finite());

    // Zero-mean with zeros denominator -> NaN
    let zeros = Series::try_new(dates, vec![0.0; 6]).unwrap();
    let zm_zero = correlation(
        &xd,
        &zeros,
        Window::Observations { size: 2, ramp: 0 },
        None,
        true,
    )
    .unwrap();
    assert!(zm_zero.values().iter().all(|v| v.is_nan()));

    // Zero-mean inner ramp + outer ramp interaction:
    // x has 5 dates starting Mar 1; y starts Mar 2 (so aligned has 4 dates starting Mar 2).
    let rx = Series::try_new(
        vec![
            date(2024, 3, 1),
            date(2024, 3, 2),
            date(2024, 3, 3),
            date(2024, 3, 4),
            date(2024, 3, 5),
        ],
        vec![0.01, 0.02, -0.01, 0.03, 0.02],
    )
    .unwrap();
    let ry = Series::try_new(
        vec![
            date(2024, 3, 2),
            date(2024, 3, 3),
            date(2024, 3, 4),
            date(2024, 3, 5),
        ],
        vec![0.01, -0.02, 0.02, 0.04],
    )
    .unwrap();
    // Observation ramp = 2: inner ramp drops aligned[0..2] (Mar 2, Mar 3); outer ramp drops x[0..2]
    // (Mar 1, Mar 2), leaving Mar 3 as NaN from inner ramp, and Mar 4 / Mar 5 finite.
    let zm_obs_ramp = correlation(
        &rx,
        &ry,
        Window::Observations { size: 2, ramp: 2 },
        None,
        true,
    )
    .unwrap();
    assert_eq!(
        zm_obs_ramp.dates(),
        &[date(2024, 3, 3), date(2024, 3, 4), date(2024, 3, 5)]
    );
    assert!(zm_obs_ramp.values()[0].is_nan());
    assert!(zm_obs_ramp.values()[1].is_finite() && zm_obs_ramp.values()[2].is_finite());

    // Calendar ramp = 2d: outer keeps >= Mar 3, inner keeps >= Mar 4 -> Mar 3 is NaN!
    let zm_cal_ramp = correlation(&rx, &ry, Window::Days { size: 2, ramp: 2 }, None, true).unwrap();
    assert_eq!(
        zm_cal_ramp.dates(),
        &[date(2024, 3, 3), date(2024, 3, 4), date(2024, 3, 5)]
    );
    assert!(zm_cal_ramp.values()[0].is_nan());
    assert!(zm_cal_ramp.values()[1].is_finite() && zm_cal_ramp.values()[2].is_finite());

    // Zero-mean Expanding on prices (aligned has len(x)-1 < len(x)) -> AnnualizationFactor
    let prices = Series::try_new(
        (1..=6).map(|d| date(2019, 1, d)).collect(),
        vec![100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792],
    )
    .unwrap();
    assert_eq!(
        correlation(
            &prices,
            &prices,
            Window::Expanding,
            Some((ReturnKind::Simple, ReturnKind::Simple)),
            true,
        ),
        Err(Error::AnnualizationFactor)
    );

    // Zero-mean inner ramp > aligned.len() (5 > 4, while 5 <= rx.len()) -> Ramp
    assert_eq!(
        correlation(
            &rx,
            &ry,
            Window::Observations { size: 2, ramp: 5 },
            None,
            true,
        ),
        Err(Error::Ramp)
    );

    // Validation, empty, oversize, non-positive prices
    let empty = Series::try_new(vec![], vec![]).unwrap();
    let non_pos =
        Series::try_new(vec![date(2024, 3, 1), date(2024, 3, 2)], vec![100.0, 0.0]).unwrap();
    assert!(correlation(
        &empty,
        &non_pos,
        Window::Observations { size: 0, ramp: 0 },
        Some((ReturnKind::Logarithmic, ReturnKind::Logarithmic)),
        true,
    )
    .unwrap()
    .values()
    .is_empty());

    assert_eq!(
        correlation(
            &rx,
            &ry,
            Window::Observations { size: 0, ramp: 0 },
            None,
            false
        ),
        Err(Error::Window)
    );
    assert_eq!(
        correlation(&rx, &ry, Window::Days { size: 0, ramp: 0 }, None, false),
        Err(Error::Window)
    );
    assert_eq!(
        correlation(
            &rx,
            &ry,
            Window::Observations { size: 2, ramp: 6 },
            None,
            false
        ),
        Err(Error::Ramp)
    );
    assert_eq!(
        correlation(
            &rx,
            &non_pos,
            Window::Observations { size: 10, ramp: 0 },
            Some((ReturnKind::Simple, ReturnKind::Logarithmic)),
            false,
        ),
        Err(Error::NonPositivePrice)
    );
    // Oversize observation window on Pearson -> empty; on empty right -> all NaN
    assert!(correlation(
        &rx,
        &ry,
        Window::Observations { size: 10, ramp: 0 },
        None,
        false,
    )
    .unwrap()
    .values()
    .is_empty());
    let empty_right = correlation(&rx, &empty, Window::Expanding, None, false).unwrap();
    assert_eq!(empty_right.dates(), rx.dates());
    assert!(empty_right.values().iter().all(|v| v.is_nan()));
    assert_eq!(
        correlation(&rx, &empty, Window::Expanding, None, true),
        Err(Error::AnnualizationFactor)
    );

    // Large offset two-pass stability for Pearson
    let shifted_x = Series::try_new(
        vec![
            date(2024, 3, 1),
            date(2024, 3, 2),
            date(2024, 3, 3),
            date(2024, 3, 4),
        ],
        vec![1e9 + 0.1, 1e9 + 0.2, 1e9 + 0.3, 1e9 + 0.5],
    )
    .unwrap();
    let shifted_y = Series::try_new(
        shifted_x.dates().to_vec(),
        vec![1e9 + 0.4, 1e9 + 0.3, 1e9 + 0.2, 1e9 + 0.0],
    )
    .unwrap();
    let shifted_corr = correlation(&shifted_x, &shifted_y, Window::Expanding, None, false).unwrap();
    assert!((shifted_corr.values()[3] - (-1.0)).abs() < 1e-6);
}

#[test]
fn beta_prices_returns_windows_masking_and_determinism() {
    let daily = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 7),
        date(2019, 1, 8),
    ];
    let px = Series::try_new(
        daily.clone(),
        vec![100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792],
    )
    .unwrap();
    let py = Series::try_new(
        daily.clone(),
        vec![100.0, 102.0, 104.0, 101.0, 100.95, 100.0],
    )
    .unwrap();

    // Default expanding window on close prices + determinism across repeated runs
    let run1 = beta(&px, &py, Window::Expanding, true).unwrap();
    let run2 = beta(&px, &py, Window::Expanding, true).unwrap();
    assert_eq!(run1.dates(), run2.dates());
    for (&a, &b) in run1.values().iter().zip(run2.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(run1.dates(), daily.as_slice());
    assert!(run1.values()[0].is_nan() && run1.values()[1].is_nan() && run1.values()[2].is_nan());
    assert!((run1.values()[3] - 0.718146).abs() < 1e-6);
    assert!((run1.values()[4] - 0.718919).abs() < 1e-6);
    assert!((run1.values()[5] - 0.572201).abs() < 1e-6);

    // Precomputed simple returns match prices=true for observation window
    let rx = returns(&px, 1, ReturnKind::Simple).unwrap();
    let ry = returns(&py, 1, ReturnKind::Simple).unwrap();
    let obs_prices = beta(&px, &py, Window::Observations { size: 2, ramp: 0 }, true).unwrap();
    let obs_rets = beta(&rx, &ry, Window::Observations { size: 2, ramp: 0 }, false).unwrap();
    assert_eq!(obs_prices.dates(), obs_rets.dates());
    for (&a, &b) in obs_prices.values().iter().zip(obs_rets.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!((obs_rets.values()[3] - 0.8255252918287954).abs() < 1e-12);
    assert!((obs_rets.values()[4] - 0.7054398925453326).abs() < 1e-12);
    assert!((obs_rets.values()[5] - (-2.24327163719368)).abs() < 1e-12);

    // Calendar-day window with ramp=0 vs default ramp=size
    let cal_0 = beta(&px, &py, Window::Days { size: 2, ramp: 0 }, true).unwrap();
    assert_eq!(cal_0.dates(), daily.as_slice());
    assert!(cal_0.values()[4].is_nan());
    assert!((cal_0.values()[3] - 0.8255252918287954).abs() < 1e-12);
    assert!((cal_0.values()[5] - (-2.24327163719368)).abs() < 1e-12);

    let cal_r = beta(&px, &py, Window::Days { size: 2, ramp: 2 }, true).unwrap();
    assert_eq!(cal_r.dates(), &daily[2..]);
    assert!(cal_r.values()[0].is_nan());
    assert!((cal_r.values()[1] - 0.8255252918287954).abs() < 1e-12);
    assert!(cal_r.values()[2].is_nan());
    assert!((cal_r.values()[3] - (-2.24327163719368)).abs() < 1e-12);

    // Offset dates: union-index masking leaves 2020-01-04 (3rd left date, 4th union date) = -0.25,
    // and missing benchmark does not pool denominator sample with pair-filtered covariance.
    let asset = Series::try_new(
        vec![1, 2, 4, 5, 7, 10, 11, 13]
            .into_iter()
            .map(|d| date(2020, 1, d))
            .collect(),
        vec![1.0, 4.0, 2.0, 9.0, 3.0, 5.0, 6.0, 2.0],
    )
    .unwrap();
    let bench = Series::try_new(
        vec![1, 3, 4, 5, 7, 8, 10, 11, 13]
            .into_iter()
            .map(|d| date(2020, 1, d))
            .collect(),
        vec![3.0, 5.0, 1.0, 9.0, 2.0, 4.0, 8.0, 3.0, 7.0],
    )
    .unwrap();
    let obs_off = beta(
        &asset,
        &bench,
        Window::Observations { size: 4, ramp: 0 },
        false,
    )
    .unwrap();
    assert_eq!(obs_off.dates(), asset.dates());
    assert!(obs_off.values()[0].is_nan() && obs_off.values()[1].is_nan());
    assert!((obs_off.values()[2] - (-0.25)).abs() < 1e-12);
    assert!((obs_off.values()[3] - 2.4).abs() < 1e-12);

    let missing_bench = Series::try_new(
        bench.dates().to_vec(),
        vec![3.0, 5.0, 1.0, 9.0, f64::NAN, 4.0, 8.0, 3.0, 7.0],
    )
    .unwrap();
    let obs_miss = beta(
        &asset,
        &missing_bench,
        Window::Observations { size: 4, ramp: 2 },
        false,
    )
    .unwrap();
    assert_eq!(obs_miss.dates(), &asset.dates()[2..]);
    assert!((obs_miss.values()[0] - (-0.25)).abs() < 1e-12);
    assert!((obs_miss.values()[2] - 1.75).abs() < 1e-12);

    // Calendar window propagates NaN through the (t - size, t] window
    let cal_miss = beta(
        &asset,
        &missing_bench,
        Window::Days { size: 5, ramp: 0 },
        false,
    )
    .unwrap();
    assert!(cal_miss.values()[..7].iter().all(|v| v.is_nan()));
    assert!((cal_miss.values()[7] - (-0.42857142857142855)).abs() < 1e-12);

    // Calendar window across leap day (2024-02-29)
    let leap_dates = vec![
        date(2024, 2, 27),
        date(2024, 2, 28),
        date(2024, 2, 29),
        date(2024, 3, 1),
        date(2024, 3, 2),
    ];
    let lx = Series::try_new(leap_dates.clone(), vec![1.0, 3.0, 2.0, 5.0, 4.0]).unwrap();
    let lb = Series::try_new(leap_dates.clone(), vec![2.0, 4.0, 1.0, 6.0, 3.0]).unwrap();
    let leap_cal = beta(&lx, &lb, Window::Days { size: 3, ramp: 2 }, false).unwrap();
    assert_eq!(leap_cal.dates(), &leap_dates[2..]);
    assert!(leap_cal.values()[0].is_nan());
    assert!(leap_cal.values()[1].is_finite() && leap_cal.values()[2].is_finite());
}

#[test]
fn beta_constant_disjoint_empty_oversize_and_errors() {
    let asset = Series::try_new(
        vec![1, 2, 4, 5, 7, 10, 11, 13]
            .into_iter()
            .map(|d| date(2020, 1, d))
            .collect(),
        vec![1.0, 4.0, 2.0, 9.0, 3.0, 5.0, 6.0, 2.0],
    )
    .unwrap();
    let bench = Series::try_new(
        vec![1, 3, 4, 5, 7, 8, 10, 11, 13]
            .into_iter()
            .map(|d| date(2020, 1, d))
            .collect(),
        vec![3.0, 5.0, 1.0, 9.0, 2.0, 4.0, 8.0, 3.0, 7.0],
    )
    .unwrap();

    // Constant benchmark (0.1, 0.01, 0.0) always produces NaN across all window types
    for c in [0.1, 0.01, 0.0] {
        let constant =
            Series::try_new(asset.dates().to_vec(), vec![c; asset.dates().len()]).unwrap();
        for w in [
            Window::Expanding,
            Window::Observations { size: 4, ramp: 0 },
            Window::Days { size: 6, ramp: 0 },
        ] {
            let res = beta(&asset, &constant, w, false).unwrap();
            assert_eq!(res.dates(), asset.dates());
            assert!(res.values().iter().all(|v| v.is_nan()));
        }
    }

    // Disjoint dates or empty benchmark produce all NaN indexed by left dates
    let disjoint = Series::try_new(
        vec![14, 15, 16, 17]
            .into_iter()
            .map(|d| date(2020, 1, d))
            .collect(),
        vec![2.0, 3.0, 1.0, 4.0],
    )
    .unwrap();
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for b in [&disjoint, &empty] {
        for w in [
            Window::Expanding,
            Window::Observations { size: 4, ramp: 0 },
            Window::Days { size: 6, ramp: 0 },
        ] {
            let res = beta(&asset, b, w, false).unwrap();
            assert_eq!(res.dates(), asset.dates());
            assert!(res.values().iter().all(|v| v.is_nan()));
        }
    }

    // Empty left series returns empty before window validation
    for w in [
        Window::Expanding,
        Window::Observations { size: 0, ramp: 0 },
        Window::Days { size: 0, ramp: 0 },
    ] {
        assert!(beta(&empty, &bench, w, true).unwrap().values().is_empty());
        assert!(beta(&empty, &empty, w, false).unwrap().values().is_empty());
    }

    // Oversize window with valid ramp -> empty; invalid ramp -> Error::Ramp first
    assert!(beta(
        &asset,
        &bench,
        Window::Observations { size: 12, ramp: 0 },
        false
    )
    .unwrap()
    .values()
    .is_empty());
    assert_eq!(
        beta(
            &asset,
            &bench,
            Window::Observations { size: 12, ramp: 12 },
            false
        ),
        Err(Error::Ramp)
    );

    // Invalid window sizes and ramps
    assert_eq!(
        beta(
            &asset,
            &bench,
            Window::Observations { size: 0, ramp: 0 },
            false
        ),
        Err(Error::Window)
    );
    assert_eq!(
        beta(&asset, &bench, Window::Days { size: 0, ramp: 0 }, false),
        Err(Error::Window)
    );
    assert_eq!(
        beta(
            &asset,
            &bench,
            Window::Observations { size: 4, ramp: 9 },
            false
        ),
        Err(Error::Ramp)
    );
}

#[test]
fn max_drawdown_windows_gaps_ramps_and_determinism() {
    let daily = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 7),
        date(2019, 1, 8),
    ];
    let prices = Series::try_new(daily.clone(), vec![1.0, 5.0, 5.0, 4.0, 4.0, 1.0]).unwrap();

    // Default expanding window + bit-exact determinism on immediate re-run
    let run1 = max_drawdown(&prices, Window::Expanding).unwrap();
    let run2 = max_drawdown(&prices, Window::Expanding).unwrap();
    assert_eq!(run1.dates(), run2.dates());
    for (&a, &b) in run1.values().iter().zip(run2.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(run1.dates(), daily.as_slice());
    let expected_default = [0.0, 0.0, 0.0, -0.2, -0.2, -0.8];
    for (&a, &e) in run1.values().iter().zip(&expected_default) {
        assert!((a - e).abs() < 1e-12);
    }

    // Trading gap comparison: 2-observation window vs 2-calendar-day window diverge on 2019-01-07
    let obs = max_drawdown(&prices, Window::Observations { size: 2, ramp: 0 }).unwrap();
    let cal = max_drawdown(&prices, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert_eq!(obs.dates(), daily.as_slice());
    assert_eq!(cal.dates(), daily.as_slice());
    assert!((obs.values()[4] - (-0.2)).abs() < 1e-12);
    assert_eq!(cal.values()[4], 0.0);
    assert!((obs.values()[5] - (-0.75)).abs() < 1e-12);
    assert!((cal.values()[5] - (-0.75)).abs() < 1e-12);

    // Observation ramp and calendar default ramp
    let obs_ramp = max_drawdown(&prices, Window::Observations { size: 2, ramp: 2 }).unwrap();
    assert_eq!(obs_ramp.dates(), &daily[2..]);
    assert!((obs_ramp.values()[2] - (-0.2)).abs() < 1e-12);
    assert!((obs_ramp.values()[3] - (-0.75)).abs() < 1e-12);

    let cal_ramp = max_drawdown(&prices, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(cal_ramp.dates(), &daily[2..]);
    assert_eq!(cal_ramp.values()[2], 0.0);
    assert!((cal_ramp.values()[3] - (-0.75)).abs() < 1e-12);

    // Missing values: NaN skipped in both stages, all-NaN window yields NaN
    let missing = Series::try_new(
        vec![
            date(2019, 1, 1),
            date(2019, 1, 2),
            date(2019, 1, 4),
            date(2019, 1, 7),
            date(2019, 1, 8),
        ],
        vec![5.0, f64::NAN, 2.0, f64::NAN, 1.0],
    )
    .unwrap();
    let miss_obs = max_drawdown(&missing, Window::Observations { size: 3, ramp: 0 }).unwrap();
    for (&a, &e) in miss_obs.values().iter().zip(&[0.0, 0.0, -0.6, -0.6, -0.6]) {
        assert!((a - e).abs() < 1e-12);
    }
    let miss_cal = max_drawdown(&missing, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert_eq!(miss_cal.values()[0], 0.0);
    assert_eq!(miss_cal.values()[1], 0.0);
    assert_eq!(miss_cal.values()[2], 0.0);
    assert!(miss_cal.values()[3].is_nan());
    assert_eq!(miss_cal.values()[4], 0.0);

    // Zero prices (0/0 -> NaN) and negative prices
    let zeros = Series::try_new(missing.dates().to_vec(), vec![0.0, 0.0, 2.0, 0.0, 1.0]).unwrap();
    let z_obs = max_drawdown(&zeros, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!(z_obs.values()[0].is_nan() && z_obs.values()[1].is_nan());
    assert_eq!(z_obs.values()[2], 0.0);
    assert_eq!(z_obs.values()[3], -1.0);
    assert_eq!(z_obs.values()[4], -1.0);

    let z_cal = max_drawdown(&zeros, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert!(z_cal.values()[0].is_nan() && z_cal.values()[1].is_nan() && z_cal.values()[3].is_nan());
    assert_eq!(z_cal.values()[2], 0.0);
    assert_eq!(z_cal.values()[4], 0.0);

    let neg =
        Series::try_new(missing.dates().to_vec(), vec![-5.0, -2.0, -9.0, -1.0, -5.0]).unwrap();
    for w in [
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        assert_eq!(max_drawdown(&neg, w).unwrap().values(), &[0.0; 5]);
    }

    // Calendar window across leap day (2024-02-29)
    let leap = Series::try_new(
        vec![
            date(2024, 2, 28),
            date(2024, 2, 29),
            date(2024, 3, 1),
            date(2024, 3, 3),
        ],
        vec![10.0, 8.0, 6.0, 5.0],
    )
    .unwrap();
    let leap_cal = max_drawdown(&leap, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(leap_cal.dates(), &[date(2024, 3, 1), date(2024, 3, 3)]);
    assert!((leap_cal.values()[0] - (-0.25)).abs() < 1e-12);
    assert_eq!(leap_cal.values()[1], 0.0);
}

#[test]
fn max_drawdown_non_finite_empty_oversize_and_errors() {
    let d4 = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
    ];

    // Negative price following zero: finite prices produce -Inf in ratio; obs/expanding skip ±Inf,
    // calendar retains ±Inf
    let neg_zero = Series::try_new(d4.clone(), vec![0.0, -1.0, -2.0, 0.0]).unwrap();
    let nz_def = max_drawdown(&neg_zero, Window::Expanding).unwrap();
    assert!(nz_def.values().iter().all(|v| v.is_nan()));
    let nz_obs = max_drawdown(&neg_zero, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert!(nz_obs.values()[0].is_nan() && nz_obs.values()[1].is_nan());
    assert_eq!(&nz_obs.values()[2..], &[1.0, 1.0]);
    let nz_cal = max_drawdown(&neg_zero, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert!(nz_cal.values()[0].is_nan());
    assert!(nz_cal.values()[1].is_infinite() && nz_cal.values()[1].is_sign_negative());
    assert!(nz_cal.values()[2].is_infinite() && nz_cal.values()[2].is_sign_negative());
    assert_eq!(nz_cal.values()[3], 1.0);
    let nz_cal_r = max_drawdown(&neg_zero, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(nz_cal_r.dates(), &d4[2..]);
    assert!(nz_cal_r.values()[0].is_infinite() && nz_cal_r.values()[0].is_sign_negative());
    assert_eq!(nz_cal_r.values()[1], 1.0);

    // Explicit +Inf: obs/expanding skip ±Inf in both stages; calendar keeps +Inf
    let inf_s = Series::try_new(d4.clone(), vec![2.0, f64::INFINITY, 1.0, -1.0]).unwrap();
    assert_eq!(
        max_drawdown(&inf_s, Window::Expanding).unwrap().values(),
        &[0.0, 0.0, -0.5, -1.5]
    );
    assert_eq!(
        max_drawdown(&inf_s, Window::Observations { size: 2, ramp: 0 })
            .unwrap()
            .values(),
        &[0.0, 0.0, 0.0, -2.0]
    );
    assert_eq!(
        max_drawdown(&inf_s, Window::Days { size: 2, ramp: 0 })
            .unwrap()
            .values(),
        &[0.0, 0.0, -1.0, -2.0]
    );
    let inf_r = max_drawdown(&inf_s, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(inf_r.dates(), &d4[2..]);
    assert_eq!(inf_r.values(), &[-1.0, -2.0]);

    // All NaN
    let all_nan = Series::try_new(d4.clone(), vec![f64::NAN; 4]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 2, ramp: 0 },
        Window::Days { size: 2, ramp: 0 },
    ] {
        let out = max_drawdown(&all_nan, w).unwrap();
        assert_eq!(out.dates(), d4.as_slice());
        assert!(out.values().iter().all(|v| v.is_nan()));
    }
    let all_nan_r = max_drawdown(&all_nan, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(all_nan_r.dates(), &d4[2..]);
    assert!(all_nan_r.values().iter().all(|v| v.is_nan()));

    // Singleton inputs
    let one_pos = Series::try_new(vec![date(2019, 1, 1)], vec![5.0]).unwrap();
    assert_eq!(
        max_drawdown(&one_pos, Window::Expanding).unwrap().values(),
        &[0.0]
    );
    assert!(
        max_drawdown(&one_pos, Window::Observations { size: 2, ramp: 0 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert_eq!(
        max_drawdown(&one_pos, Window::Days { size: 2, ramp: 0 })
            .unwrap()
            .values(),
        &[0.0]
    );
    assert!(max_drawdown(&one_pos, Window::Days { size: 2, ramp: 2 })
        .unwrap()
        .values()
        .is_empty());
    let one_zero = Series::try_new(vec![date(2019, 1, 1)], vec![0.0]).unwrap();
    assert!(max_drawdown(&one_zero, Window::Expanding).unwrap().values()[0].is_nan());
    let one_neg = Series::try_new(vec![date(2019, 1, 1)], vec![-5.0]).unwrap();
    assert_eq!(
        max_drawdown(&one_neg, Window::Expanding).unwrap().values(),
        &[0.0]
    );

    // Empty inputs: positive calendar ramp maps upstream IndexError -> Error::Ramp; others empty
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 2, ramp: 0 },
        Window::Observations { size: 0, ramp: 0 },
        Window::Days { size: 2, ramp: 0 },
        Window::Days { size: 0, ramp: 0 },
    ] {
        assert!(max_drawdown(&empty, w).unwrap().values().is_empty());
    }
    assert_eq!(
        max_drawdown(&empty, Window::Days { size: 2, ramp: 2 }),
        Err(Error::Ramp)
    );

    // Oversize windows, ramp order, and zero-size window errors
    let sample = Series::try_new(d4, vec![1.0, 5.0, 4.0, 2.0]).unwrap();
    assert!(
        max_drawdown(&sample, Window::Observations { size: 10, ramp: 0 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert!(
        max_drawdown(&sample, Window::Observations { size: 4, ramp: 4 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert_eq!(
        max_drawdown(&sample, Window::Observations { size: 10, ramp: 10 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        max_drawdown(&sample, Window::Observations { size: 2, ramp: 5 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        max_drawdown(&sample, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        max_drawdown(&sample, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
}

#[test]
fn max_drawdown_signed_zero_calendar_ties() {
    let dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 2, 29),
        date(2020, 3, 1),
    ];

    // [0.0, -0.0, -1.0, -2.0]: calendar peak keeps the last tied zero (-0.0) -> +Inf
    let pos_neg = Series::try_new(dates.clone(), vec![0.0, -0.0, -1.0, -2.0]).unwrap();
    let pn_3d_a = max_drawdown(&pos_neg, Window::Days { size: 3, ramp: 0 }).unwrap();
    let pn_3d_b = max_drawdown(&pos_neg, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert_eq!(pn_3d_a.dates(), dates.as_slice());
    assert_eq!(pn_3d_a.dates(), pn_3d_b.dates());
    for (&a, &b) in pn_3d_a.values().iter().zip(pn_3d_b.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(pn_3d_a.values()[0].is_nan() && pn_3d_a.values()[1].is_nan());
    assert_eq!(pn_3d_a.values()[2], f64::INFINITY);
    assert_eq!(pn_3d_a.values()[3], f64::INFINITY);

    let pn_2d = max_drawdown(&pos_neg, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert!(pn_2d.values()[0].is_nan() && pn_2d.values()[1].is_nan());
    assert_eq!(pn_2d.values()[2], f64::INFINITY);
    assert_eq!(pn_2d.values()[3], 1.0);

    let pn_obs2 = max_drawdown(&pos_neg, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert!(
        pn_obs2.values()[0].is_nan()
            && pn_obs2.values()[1].is_nan()
            && pn_obs2.values()[2].is_nan()
    );
    assert_eq!(pn_obs2.values()[3], 1.0);

    let pn_3d_ramp = max_drawdown(&pos_neg, Window::Days { size: 3, ramp: 3 }).unwrap();
    assert_eq!(pn_3d_ramp.dates(), &[date(2020, 3, 1)]);
    assert_eq!(pn_3d_ramp.values(), &[f64::INFINITY]);

    // [-0.0, 0.0, -1.0, -2.0]: calendar peak keeps the last tied zero (+0.0) -> -Inf
    let neg_pos = Series::try_new(dates.clone(), vec![-0.0, 0.0, -1.0, -2.0]).unwrap();
    let np_3d_a = max_drawdown(&neg_pos, Window::Days { size: 3, ramp: 0 }).unwrap();
    let np_3d_b = max_drawdown(&neg_pos, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert_eq!(np_3d_a.dates(), dates.as_slice());
    assert_eq!(np_3d_a.dates(), np_3d_b.dates());
    for (&a, &b) in np_3d_a.values().iter().zip(np_3d_b.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(np_3d_a.values()[0].is_nan() && np_3d_a.values()[1].is_nan());
    assert_eq!(np_3d_a.values()[2], f64::NEG_INFINITY);
    assert_eq!(np_3d_a.values()[3], f64::NEG_INFINITY);

    let np_2d = max_drawdown(&neg_pos, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert!(np_2d.values()[0].is_nan() && np_2d.values()[1].is_nan());
    assert_eq!(np_2d.values()[2], f64::NEG_INFINITY);
    assert_eq!(np_2d.values()[3], f64::NEG_INFINITY);

    let np_obs2 = max_drawdown(&neg_pos, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert!(
        np_obs2.values()[0].is_nan()
            && np_obs2.values()[1].is_nan()
            && np_obs2.values()[2].is_nan()
    );
    assert_eq!(np_obs2.values()[3], 1.0);

    let np_3d_ramp = max_drawdown(&neg_pos, Window::Days { size: 3, ramp: 3 }).unwrap();
    assert_eq!(np_3d_ramp.dates(), &[date(2020, 3, 1)]);
    assert_eq!(np_3d_ramp.values(), &[f64::NEG_INFINITY]);

    // [-0.0, -0.0, -1.0, -2.0]: both zeros negative -> +Inf in 3d and 2d, skipped in obs 2
    let neg_neg = Series::try_new(dates.clone(), vec![-0.0, -0.0, -1.0, -2.0]).unwrap();
    let nn_3d = max_drawdown(&neg_neg, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert!(nn_3d.values()[0].is_nan() && nn_3d.values()[1].is_nan());
    assert_eq!(nn_3d.values()[2], f64::INFINITY);
    assert_eq!(nn_3d.values()[3], f64::INFINITY);
    let nn_obs2 = max_drawdown(&neg_neg, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert!(
        nn_obs2.values()[0].is_nan()
            && nn_obs2.values()[1].is_nan()
            && nn_obs2.values()[2].is_nan()
    );
    assert_eq!(nn_obs2.values()[3], 1.0);

    // NaN interleaved between tied signed zeros and negative price
    let nan_pos_neg = Series::try_new(dates.clone(), vec![0.0, -0.0, f64::NAN, -1.0]).unwrap();
    let npn_4d = max_drawdown(&nan_pos_neg, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert!(
        npn_4d.values()[0].is_nan() && npn_4d.values()[1].is_nan() && npn_4d.values()[2].is_nan()
    );
    assert_eq!(npn_4d.values()[3], f64::INFINITY);

    let nan_neg_pos = Series::try_new(dates, vec![-0.0, 0.0, f64::NAN, -1.0]).unwrap();
    let nnp_4d = max_drawdown(&nan_neg_pos, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert!(
        nnp_4d.values()[0].is_nan() && nnp_4d.values()[1].is_nan() && nnp_4d.values()[2].is_nan()
    );
    assert_eq!(nnp_4d.values()[3], f64::NEG_INFINITY);
}

#[test]
fn ema_irregular_dates_default_and_custom_betas() {
    let daily_dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 5),
        date(2019, 1, 6),
    ];
    let irregular_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 3, 2),
        date(2020, 3, 7),
        date(2020, 3, 14),
        date(2020, 3, 23),
    ];
    let raw = vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0];
    let daily = Series::try_new(daily_dates, raw.clone()).unwrap();
    let irregular = Series::try_new(irregular_dates.clone(), raw.clone()).unwrap();

    // Default beta = 0.75; row spacing yields identical values across daily and irregular dates
    let def_daily = exponential_moving_average(&daily, None).unwrap();
    let def_irreg = exponential_moving_average(&irregular, None).unwrap();
    let def_explicit = exponential_moving_average(&irregular, Some(0.75)).unwrap();
    assert_eq!(def_irreg.dates(), irregular_dates.as_slice());
    assert_eq!(def_irreg.values(), def_daily.values());
    assert_eq!(def_irreg.values(), def_explicit.values());
    let expected_default = [3.0, 2.75, 2.8125, 2.359375, 2.51953125, 3.3896484375];
    for (&actual, &expected) in def_irreg.values().iter().zip(&expected_default) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // beta = 0.6 preserves dates and matches pinned Python values
    let b06 = exponential_moving_average(&irregular, Some(0.6)).unwrap();
    assert_eq!(b06.dates(), irregular_dates.as_slice());
    let expected_06 = [3.0, 2.6, 2.76, 2.056, 2.4336, 3.86016];
    for (&actual, &expected) in b06.values().iter().zip(&expected_06) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // beta = 0.0 and -0.0 return the input values unchanged on finite series
    for zero_beta in [0.0, -0.0] {
        let b0 = exponential_moving_average(&irregular, Some(zero_beta)).unwrap();
        assert_eq!(b0.dates(), irregular_dates.as_slice());
        assert_eq!(b0.values(), raw.as_slice());
    }

    // Valid upper boundary near 1.0 (1.0 - 2^-53)
    let gap_dates = irregular_dates[..5].to_vec();
    let gaps = Series::try_new(
        gap_dates.clone(),
        vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN],
    )
    .unwrap();
    let near_one = exponential_moving_average(&gaps, Some(0.9999999999999999)).unwrap();
    assert_eq!(near_one.dates(), gap_dates.as_slice());
    assert_eq!(near_one.values()[0], 1.0);
    assert_eq!(near_one.values()[2], 1.0);
    assert!((near_one.values()[3] - (1.0 + f64::EPSILON)).abs() <= 1e-15);
    assert_eq!(
        near_one.values()[4].to_bits(),
        near_one.values()[3].to_bits()
    );
}

#[test]
fn ema_missing_infinities_signed_zero_and_determinism() {
    let dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 3, 2),
        date(2020, 3, 7),
        date(2020, 3, 14),
    ];

    // Leading NaNs stay NaN until the first finite observation
    let leading = Series::try_new(dates[..4].to_vec(), vec![f64::NAN, f64::NAN, 2.0, 6.0]).unwrap();
    let lead_def = exponential_moving_average(&leading, None).unwrap();
    assert!(lead_def.values()[0].is_nan() && lead_def.values()[1].is_nan());
    assert_eq!(&lead_def.values()[2..], &[2.0, 3.0]);
    let lead_0 = exponential_moving_average(&leading, Some(0.0)).unwrap();
    assert!(lead_0.values()[0].is_nan() && lead_0.values()[1].is_nan());
    assert_eq!(&lead_0.values()[2..], &[2.0, 6.0]);

    // Missing rows (NaN and ±Inf) carry forward prior EMA while decaying old weight across gaps,
    // and consecutive runs produce bit-exact identical outputs
    let mixed = Series::try_new(
        dates.clone(),
        vec![f64::INFINITY, 3.0, f64::NEG_INFINITY, 6.0, f64::NAN],
    )
    .unwrap();
    let run1 = exponential_moving_average(&mixed, None).unwrap();
    let run2 = exponential_moving_average(&mixed, None).unwrap();
    assert_eq!(run1.dates(), run2.dates());
    assert_eq!(run1.dates(), dates.as_slice());
    for (&a, &b) in run1.values().iter().zip(run2.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(run1.values()[0].is_nan());
    assert_eq!(run1.values()[1], 3.0);
    assert_eq!(run1.values()[2], 3.0);
    assert!((run1.values()[3] - (51.0 / 13.0)).abs() < 1e-12);
    assert_eq!(run1.values()[4].to_bits(), run1.values()[3].to_bits());

    let mixed_b0 = exponential_moving_average(&mixed, Some(0.0)).unwrap();
    assert!(mixed_b0.values()[0].is_nan());
    assert_eq!(&mixed_b0.values()[1..], &[3.0, 3.0, 6.0, 6.0]);

    // Signed-zero preservation when equal to previous average (even with beta = 0.0)
    let neg_first = Series::try_new(dates[..4].to_vec(), vec![-0.0, 0.0, f64::NAN, -0.0]).unwrap();
    for b in [None, Some(0.0)] {
        let out = exponential_moving_average(&neg_first, b).unwrap();
        assert_eq!(out.dates(), &dates[..4]);
        assert!(out
            .values()
            .iter()
            .all(|v| *v == 0.0 && v.is_sign_negative()));
    }

    let pos_first = Series::try_new(dates[..4].to_vec(), vec![0.0, -0.0, f64::NAN, 0.0]).unwrap();
    for b in [None, Some(0.0)] {
        let out = exponential_moving_average(&pos_first, b).unwrap();
        assert!(out
            .values()
            .iter()
            .all(|v| *v == 0.0 && v.is_sign_positive()));
    }

    // Transition from non-zero to -0.0 at beta = 0.0 follows IEEE-754 weighted sum sign
    let from_pos = Series::try_new(dates[..2].to_vec(), vec![1.0, -0.0]).unwrap();
    let out_pos = exponential_moving_average(&from_pos, Some(0.0)).unwrap();
    assert!(out_pos.values()[1] == 0.0 && out_pos.values()[1].is_sign_positive());

    let from_neg = Series::try_new(dates[..2].to_vec(), vec![-1.0, -0.0]).unwrap();
    let out_neg = exponential_moving_average(&from_neg, Some(0.0)).unwrap();
    assert!(out_neg.values()[1] == 0.0 && out_neg.values()[1].is_sign_negative());
}

#[test]
fn ema_empty_singleton_all_missing_and_beta_errors() {
    let d1 = vec![date(2020, 2, 27)];
    let empty = Series::try_new(vec![], vec![]).unwrap();

    // Valid beta on empty series -> empty series
    for b in [
        None,
        Some(0.0),
        Some(-0.0),
        Some(0.6),
        Some(0.9999999999999999),
    ] {
        let out = exponential_moving_average(&empty, b).unwrap();
        assert!(out.dates().is_empty() && out.values().is_empty());
    }

    // Singleton finite, -0.0, NaN, ±Inf, and all-missing
    let one = Series::try_new(d1.clone(), vec![42.0]).unwrap();
    assert_eq!(
        exponential_moving_average(&one, None).unwrap().values(),
        &[42.0]
    );

    let one_neg0 = Series::try_new(d1.clone(), vec![-0.0]).unwrap();
    let out_neg0 = exponential_moving_average(&one_neg0, None).unwrap();
    assert!(out_neg0.values()[0] == 0.0 && out_neg0.values()[0].is_sign_negative());

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let one_bad = Series::try_new(d1.clone(), vec![bad]).unwrap();
        assert!(exponential_moving_average(&one_bad, None).unwrap().values()[0].is_nan());
    }

    let all_missing = Series::try_new(
        vec![date(2020, 2, 27), date(2020, 2, 28), date(2020, 3, 2)],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    )
    .unwrap();
    let all_out = exponential_moving_average(&all_missing, None).unwrap();
    assert_eq!(all_out.dates(), all_missing.dates());
    assert!(all_out.values().iter().all(|v| v.is_nan()));

    // Invalid beta values consistently return Err(Error::Beta) on both non-empty and empty inputs
    let multi = Series::try_new(
        vec![date(2020, 2, 27), date(2020, 2, 28), date(2020, 3, 2)],
        vec![2.0, 3.0, 4.0],
    )
    .unwrap();
    for invalid_beta in [
        -0.1,
        -f64::MIN_POSITIVE,
        -f64::from_bits(1),
        1.0,
        1.1,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        assert_eq!(
            exponential_moving_average(&one, Some(invalid_beta)),
            Err(Error::Beta)
        );
        assert_eq!(
            exponential_moving_average(&multi, Some(invalid_beta)),
            Err(Error::Beta)
        );
        assert_eq!(
            exponential_moving_average(&empty, Some(invalid_beta)),
            Err(Error::Beta)
        );
    }
}

#[test]
fn smma_expanding_and_observation_windows() {
    let dates: Vec<_> = (1..=6).map(|d| date(2019, 1, d)).collect();
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // Default expanding window (normalized size = len(x) = 6, ramp = 0)
    let def = smoothed_moving_average(&x, Window::Expanding).unwrap();
    assert_eq!(def.dates(), dates.as_slice());
    let expected_def = [
        3.0,
        17.0 / 6.0,
        103.0 / 36.0,
        551.0 / 216.0,
        3403.0 / 1296.0,
        24791.0 / 7776.0,
    ];
    for (&actual, &expected) in def.values().iter().zip(&expected_def) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // Observation window size 1 -> identity on finite series
    let obs1 = smoothed_moving_average(&x, Window::Observations { size: 1, ramp: 0 }).unwrap();
    assert_eq!(obs1.dates(), dates.as_slice());
    assert_eq!(obs1.values(), &[3.0, 2.0, 3.0, 1.0, 3.0, 6.0]);

    // Observation window size 2, ramp 0 vs ramp 2
    let obs2_r0 = smoothed_moving_average(&x, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(obs2_r0.dates(), dates.as_slice());
    assert_eq!(obs2_r0.values(), &[3.0, 2.5, 2.75, 1.875, 2.4375, 4.21875]);

    let obs2_r2 = smoothed_moving_average(&x, Window::Observations { size: 2, ramp: 2 }).unwrap();
    assert_eq!(obs2_r2.dates(), &dates[2..]);
    assert_eq!(obs2_r2.values(), &[2.5, 1.75, 2.375, 4.1875]);

    // Observation window size 3, ramp 2: initial mean is mean([3, 2, 3]) = 8/3
    let obs3_r2 = smoothed_moving_average(&x, Window::Observations { size: 3, ramp: 2 }).unwrap();
    assert_eq!(obs3_r2.dates(), &dates[2..]);
    let expected_obs3 = [8.0 / 3.0, 19.0 / 9.0, 65.0 / 27.0, 292.0 / 81.0];
    for (&actual, &expected) in obs3_r2.values().iter().zip(&expected_obs3) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // Ramp == len -> empty series
    let ramp_eq = smoothed_moving_average(&x, Window::Observations { size: 2, ramp: 6 }).unwrap();
    assert!(ramp_eq.dates().is_empty() && ramp_eq.values().is_empty());
}

#[test]
fn smma_calendar_windows_irregular_leap_day_and_post_ramp_slicing() {
    let regular_dates: Vec<_> = (1..=6).map(|d| date(2019, 1, d)).collect();
    let regular =
        Series::try_new(regular_dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // 2d with default ramp (2) vs ramp 0 on regular daily dates
    let reg_2d = smoothed_moving_average(&regular, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(reg_2d.dates(), &regular_dates[2..]);
    assert_eq!(reg_2d.values(), &[2.5, 1.75, 2.375, 4.1875]);

    let reg_2d_r0 = smoothed_moving_average(&regular, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert_eq!(reg_2d_r0.dates(), regular_dates.as_slice());
    assert_eq!(
        reg_2d_r0.values(),
        &[3.0, 2.5, 2.75, 1.875, 2.4375, 4.21875]
    );

    // 4d with ramp 2d on dense dates: initial value uses full x ([3, 2, 3] -> 8/3),
    // while recurrence counts observations inside post-ramp sliced x (so Jan 4 has n = 2, not 4).
    let reg_4d_r2 = smoothed_moving_average(&regular, Window::Days { size: 4, ramp: 2 }).unwrap();
    assert_eq!(reg_4d_r2.dates(), &regular_dates[2..]);
    let s0 = 8.0 / 3.0;
    let s1 = (1.0 * s0 + 1.0) / 2.0; // {Jan 3, Jan 4} -> n = 2
    let s2 = (2.0 * s1 + 3.0) / 3.0; // {Jan 3, Jan 4, Jan 5} -> n = 3
    let s3 = (3.0 * s2 + 6.0) / 4.0; // {Jan 3, Jan 4, Jan 5, Jan 6} -> n = 4
    for (&actual, &expected) in reg_4d_r2.values().iter().zip(&[s0, s1, s2, s3]) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // Irregular spacing crossing leap day 2020-02-29
    let irreg_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 3, 2),
        date(2020, 3, 7),
        date(2020, 3, 14),
        date(2020, 3, 23),
    ];
    let irreg = Series::try_new(irreg_dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    let irreg_4d = smoothed_moving_average(&irreg, Window::Days { size: 4, ramp: 4 }).unwrap();
    assert_eq!(irreg_4d.dates(), &irreg_dates[2..]);
    assert_eq!(irreg_4d.values(), &[2.5, 1.0, 3.0, 6.0]);

    let irreg_4d_r0 = smoothed_moving_average(&irreg, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert_eq!(irreg_4d_r0.dates(), irreg_dates.as_slice());
    assert_eq!(irreg_4d_r0.values(), &[3.0, 2.5, 2.75, 1.0, 3.0, 6.0]);

    let irreg_2d = smoothed_moving_average(&irreg, Window::Days { size: 2, ramp: 2 }).unwrap();
    assert_eq!(irreg_2d.dates(), &irreg_dates[2..]);
    assert_eq!(irreg_2d.values(), &[3.0, 1.0, 3.0, 6.0]);

    let irreg_2d_r0 = smoothed_moving_average(&irreg, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert_eq!(irreg_2d_r0.dates(), irreg_dates.as_slice());
    assert_eq!(irreg_2d_r0.values(), &[3.0, 2.5, 3.0, 1.0, 3.0, 6.0]);

    // 4d with ramp 1d: Feb 28 survives ramp, so Mar 2 sees {Feb 28, Mar 2} (n = 2)
    let irreg_4d_r1 = smoothed_moving_average(&irreg, Window::Days { size: 4, ramp: 1 }).unwrap();
    assert_eq!(irreg_4d_r1.dates(), &irreg_dates[1..]);
    assert_eq!(irreg_4d_r1.values(), &[2.5, 2.75, 1.0, 3.0, 6.0]);

    // Extreme calendar ranges handled via integer ordinals (beyond pandas DateOffset limits)
    let ancient =
        Series::try_new(vec![date(1677, 9, 22), date(1677, 9, 23)], vec![2.0, 4.0]).unwrap();
    let anc_out = smoothed_moving_average(&ancient, Window::Days { size: 2, ramp: 0 }).unwrap();
    assert_eq!(anc_out.dates(), ancient.dates());
    assert_eq!(anc_out.values(), &[2.0, 3.0]);
    let huge_ramp = smoothed_moving_average(
        &regular,
        Window::Days {
            size: 1_000_000,
            ramp: 1_000_000,
        },
    )
    .unwrap();
    assert!(huge_ramp.dates().is_empty() && huge_ramp.values().is_empty());
}

#[test]
fn smma_nan_inf_signed_zero_oversize_and_determinism() {
    let dates: Vec<_> = (1..=6).map(|d| date(2019, 1, d)).collect();

    // NaN propagation: leading, mid, all-NaN, and NaN skipped in initial mean window
    let nan_lead = Series::try_new(dates.clone(), vec![f64::NAN, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let nl1 =
        smoothed_moving_average(&nan_lead, Window::Observations { size: 2, ramp: 0 }).unwrap();
    let nl2 =
        smoothed_moving_average(&nan_lead, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(nl1.dates(), nl2.dates());
    assert!(nl1.values().iter().all(|v| v.is_nan()));
    assert!(nl1
        .values()
        .iter()
        .zip(nl2.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));

    let nan_mid = Series::try_new(dates.clone(), vec![2.0, 4.0, f64::NAN, 1.0, 3.0, 6.0]).unwrap();
    let nm = smoothed_moving_average(&nan_mid, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(&nm.values()[..2], &[2.0, 3.0]);
    assert!(nm.values()[2..].iter().all(|v| v.is_nan()));

    let nan_init = Series::try_new(dates[..4].to_vec(), vec![f64::NAN, 2.0, 4.0, 6.0]).unwrap();
    let ni = smoothed_moving_average(&nan_init, Window::Observations { size: 3, ramp: 2 }).unwrap();
    assert_eq!(ni.dates(), &dates[2..4]);
    assert_eq!(ni.values(), &[3.0, 4.0]);

    // ±Inf: skipped in initial mean window (matching pandas rolling().mean()), propagated in loop
    let inf_lead =
        Series::try_new(dates[..4].to_vec(), vec![f64::INFINITY, 2.0, 4.0, 6.0]).unwrap();
    let il_r0 =
        smoothed_moving_average(&inf_lead, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert!(il_r0.values().iter().all(|v| v.is_nan()));
    let il_r1 =
        smoothed_moving_average(&inf_lead, Window::Observations { size: 2, ramp: 1 }).unwrap();
    assert_eq!(il_r1.dates(), &dates[1..4]);
    assert_eq!(il_r1.values(), &[2.0, 3.0, 4.5]);
    let il_2d_r1 = smoothed_moving_average(&inf_lead, Window::Days { size: 2, ramp: 1 }).unwrap();
    assert_eq!(il_2d_r1.dates(), &dates[1..4]);
    assert_eq!(il_2d_r1.values(), &[2.0, 3.0, 4.5]);

    let inf_mid =
        Series::try_new(dates.clone(), vec![2.0, 4.0, f64::INFINITY, 1.0, 3.0, 6.0]).unwrap();
    let im1 = smoothed_moving_average(&inf_mid, Window::Observations { size: 2, ramp: 0 }).unwrap();
    let im2 = smoothed_moving_average(&inf_mid, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(im1.dates(), im2.dates());
    assert!(im1
        .values()
        .iter()
        .zip(im2.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(
        im1.values(),
        &[
            2.0,
            3.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY
        ]
    );

    // Opposite-sign infinity and 0 * inf produce NaN
    let inf_opp = Series::try_new(
        dates[..4].to_vec(),
        vec![2.0, f64::INFINITY, f64::NEG_INFINITY, 6.0],
    )
    .unwrap();
    let io = smoothed_moving_average(&inf_opp, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(io.values()[0], 2.0);
    assert_eq!(io.values()[1], f64::INFINITY);
    assert!(io.values()[2].is_nan() && io.values()[3].is_nan());

    let inf_w1 = Series::try_new(dates[..3].to_vec(), vec![2.0, f64::INFINITY, 6.0]).unwrap();
    let iw1 = smoothed_moving_average(&inf_w1, Window::Observations { size: 1, ramp: 0 }).unwrap();
    assert_eq!(iw1.values()[0], 2.0);
    assert_eq!(iw1.values()[1], f64::INFINITY);
    assert!(iw1.values()[2].is_nan());

    // Signed zero: [-0.0, 0.0, -0.0, -0.0] -> [-0.0, +0.0, +0.0, +0.0] deterministically
    let neg_zero = Series::try_new(dates[..4].to_vec(), vec![-0.0, 0.0, -0.0, -0.0]).unwrap();
    let nz1 =
        smoothed_moving_average(&neg_zero, Window::Observations { size: 2, ramp: 0 }).unwrap();
    let nz2 =
        smoothed_moving_average(&neg_zero, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(nz1.dates(), nz2.dates());
    assert!(nz1
        .values()
        .iter()
        .zip(nz2.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(nz1.values()[0] == 0.0 && nz1.values()[0].is_sign_negative());
    assert!(nz1.values()[1..]
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));

    let all_neg_zero = Series::try_new(dates[..4].to_vec(), vec![-0.0; 4]).unwrap();
    let anz =
        smoothed_moving_average(&all_neg_zero, Window::Observations { size: 2, ramp: 2 }).unwrap();
    assert_eq!(anz.dates(), &dates[2..4]);
    assert!(anz
        .values()
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_negative()));

    // Oversize windows (size > len with valid ramp) -> empty, deterministic across re-runs
    let full = Series::try_new(dates, vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let ov1 = smoothed_moving_average(&full, Window::Observations { size: 10, ramp: 0 }).unwrap();
    let ov2 = smoothed_moving_average(&full, Window::Observations { size: 10, ramp: 0 }).unwrap();
    assert!(ov1.dates().is_empty() && ov2.dates().is_empty());
    let ov_days = smoothed_moving_average(
        &full,
        Window::Days {
            size: 100,
            ramp: 100,
        },
    )
    .unwrap();
    assert!(ov_days.dates().is_empty() && ov_days.values().is_empty());
}

#[test]
fn smma_empty_singleton_and_invalid_window_errors() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    assert!(smoothed_moving_average(&empty, Window::Expanding)
        .unwrap()
        .values()
        .is_empty());
    assert!(
        smoothed_moving_average(&empty, Window::Observations { size: 2, ramp: 0 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert!(
        smoothed_moving_average(&empty, Window::Observations { size: 2, ramp: 2 })
            .unwrap()
            .values()
            .is_empty()
    );
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

    // Singleton series
    let single = Series::try_new(vec![date(2019, 1, 1)], vec![5.0]).unwrap();
    let s_def = smoothed_moving_average(&single, Window::Expanding).unwrap();
    assert_eq!(s_def.dates(), &[date(2019, 1, 1)]);
    assert_eq!(s_def.values(), &[5.0]);
    assert!(
        smoothed_moving_average(&single, Window::Observations { size: 2, ramp: 0 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert_eq!(
        smoothed_moving_average(&single, Window::Observations { size: 2, ramp: 2 }),
        Err(Error::Ramp)
    );
    assert!(
        smoothed_moving_average(&single, Window::Days { size: 2, ramp: 2 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert_eq!(
        smoothed_moving_average(&single, Window::Days { size: 2, ramp: 0 })
            .unwrap()
            .values(),
        &[5.0]
    );

    // Non-empty invalid window / ramp errors
    let multi = Series::try_new(
        (1..=6).map(|d| date(2019, 1, d)).collect(),
        vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0],
    )
    .unwrap();
    assert_eq!(
        smoothed_moving_average(&multi, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        smoothed_moving_average(&multi, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        smoothed_moving_average(&multi, Window::Days { size: 0, ramp: 2 }),
        Err(Error::Window)
    );
    assert_eq!(
        smoothed_moving_average(&multi, Window::Observations { size: 10, ramp: 10 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        smoothed_moving_average(&multi, Window::Observations { size: 2, ramp: 7 }),
        Err(Error::Ramp)
    );
}

#[test]
fn rsi_default_and_observation_windows() {
    let spx_dates = vec![
        date(2020, 1, 2),
        date(2020, 1, 3),
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
        date(2020, 1, 9),
        date(2020, 1, 10),
        date(2020, 1, 13),
        date(2020, 1, 14),
        date(2020, 1, 15),
        date(2020, 1, 16),
        date(2020, 1, 17),
        date(2020, 1, 21),
        date(2020, 1, 22),
        date(2020, 1, 23),
        date(2020, 1, 24),
        date(2020, 1, 27),
        date(2020, 1, 28),
        date(2020, 1, 29),
        date(2020, 1, 30),
        date(2020, 1, 31),
        date(2020, 2, 3),
    ];
    let spx_values = vec![
        3257.8501, 3234.8501, 3246.28, 3237.1799, 3253.05, 3274.7, 3265.3501, 3288.1299, 3283.1499,
        3289.29, 3316.8101, 3329.6201, 3320.79, 3321.75, 3325.54, 3295.47, 3243.6299, 3276.24,
        3273.3999, 3283.6599, 3225.52, 3248.9199,
    ];
    let spx = Series::try_new(spx_dates.clone(), spx_values).unwrap();

    // Default None equals explicit Observations { size: 14, ramp: 14 }
    let def = relative_strength_index(&spx, None).unwrap();
    let exp14 =
        relative_strength_index(&spx, Some(Window::Observations { size: 14, ramp: 14 })).unwrap();
    assert_eq!(def.dates(), &spx_dates[15..]);
    assert_eq!(def.dates(), exp14.dates());
    assert!(def
        .values()
        .iter()
        .zip(exp14.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    let expected_spx = [
        66.35898836410378,
        50.99376669117847,
        57.63855067827995,
        56.91474740753822,
        58.92161739096986,
        45.88013675902517,
        50.617629713259795,
    ];
    for (&actual, &expected) in def.values().iter().zip(&expected_spx) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // Specified observation windows (size 3 ramp 3 vs ramp 0, size 1 ramp 0)
    let dates10: Vec<_> = (1..=10).map(|d| date(2020, 1, d)).collect();
    let mixed = Series::try_new(
        dates10.clone(),
        vec![10.0, 12.0, 11.0, 13.0, 13.0, 9.0, 10.5, 14.0, 12.0, 15.0],
    )
    .unwrap();
    let obs3_r3 =
        relative_strength_index(&mixed, Some(Window::Observations { size: 3, ramp: 3 })).unwrap();
    assert_eq!(obs3_r3.dates(), &dates10[4..]);
    assert!((obs3_r3.values()[0] - (200.0 / 3.0)).abs() < 1e-12);
    assert!((obs3_r3.values()[5] - 70.00547345374932).abs() < 1e-12);

    let obs3_r0 =
        relative_strength_index(&mixed, Some(Window::Observations { size: 3, ramp: 0 })).unwrap();
    assert_eq!(obs3_r0.dates(), &dates10[1..]);
    assert_eq!(obs3_r0.values()[0], 100.0);
    assert_eq!(obs3_r0.values()[1], 80.0);
    assert_eq!(obs3_r0.values()[2], 87.5);
    assert_eq!(obs3_r0.values()[3], 87.5);

    let obs1 = relative_strength_index(
        &Series::try_new(dates10[..5].to_vec(), vec![10.0, 12.0, 11.0, 13.0, 13.0]).unwrap(),
        Some(Window::Observations { size: 1, ramp: 0 }),
    )
    .unwrap();
    assert_eq!(obs1.dates(), &dates10[1..5]);
    assert_eq!(obs1.values(), &[100.0, 0.0, 100.0, 100.0]);
}

#[test]
fn rsi_calendar_windows_and_leap_day_irregular_dates() {
    let dates10: Vec<_> = (1..=10).map(|d| date(2020, 1, d)).collect();
    let inc = Series::try_new(dates10.clone(), (1..=10).map(f64::from).collect()).unwrap();
    let inc_3d = relative_strength_index(&inc, Some(Window::Days { size: 3, ramp: 3 })).unwrap();
    assert_eq!(inc_3d.dates(), &dates10[4..]);
    assert_eq!(inc_3d.values(), &[100.0; 6]);

    let mixed = Series::try_new(
        dates10.clone(),
        vec![10.0, 12.0, 11.0, 13.0, 13.0, 9.0, 10.5, 14.0, 12.0, 15.0],
    )
    .unwrap();
    let mix_3d = relative_strength_index(&mixed, Some(Window::Days { size: 3, ramp: 3 })).unwrap();
    assert_eq!(mix_3d.dates(), &dates10[4..]);
    assert!((mix_3d.values()[0] - (200.0 / 3.0)).abs() < 1e-12);
    assert!((mix_3d.values()[5] - 66.5086887835703).abs() < 1e-12);

    // Irregular dates crossing leap day 2020-02-29
    let irreg_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 3, 2),
        date(2020, 3, 4),
        date(2020, 3, 7),
        date(2020, 3, 10),
        date(2020, 3, 14),
        date(2020, 3, 20),
    ];
    let irreg = Series::try_new(
        irreg_dates.clone(),
        vec![10.0, 12.0, 11.0, 14.0, 12.0, 12.0, 15.0, 9.0],
    )
    .unwrap();

    let irreg_obs2 =
        relative_strength_index(&irreg, Some(Window::Observations { size: 2, ramp: 2 })).unwrap();
    assert_eq!(irreg_obs2.dates(), &irreg_dates[3..]);
    assert_eq!(irreg_obs2.values(), &[75.0, 37.5, 37.5, 84.375, 21.09375]);

    let irreg_4d =
        relative_strength_index(&irreg, Some(Window::Days { size: 4, ramp: 4 })).unwrap();
    assert_eq!(irreg_4d.dates(), &irreg_dates[3..]);
    assert_eq!(irreg_4d.values(), &[75.0, 37.5, 37.5, 100.0, 0.0]);

    let irreg_4d_r0 =
        relative_strength_index(&irreg, Some(Window::Days { size: 4, ramp: 0 })).unwrap();
    assert_eq!(irreg_4d_r0.dates(), &irreg_dates[1..]);
    let expected_4d_r0 = [
        100.0,
        200.0 / 3.0,
        800.0 / 9.0,
        800.0 / 17.0,
        800.0 / 17.0,
        100.0,
        0.0,
    ];
    for (&actual, &expected) in irreg_4d_r0.values().iter().zip(&expected_4d_r0) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // Extreme calendar dates handled via integer ordinals (beyond pandas DateOffset range)
    let ancient = Series::try_new(
        vec![date(1677, 9, 22), date(1677, 9, 23), date(1677, 9, 24)],
        vec![10.0, 12.0, 11.0],
    )
    .unwrap();
    let anc_out =
        relative_strength_index(&ancient, Some(Window::Days { size: 2, ramp: 0 })).unwrap();
    assert_eq!(anc_out.dates(), &[date(1677, 9, 23), date(1677, 9, 24)]);
    assert!((anc_out.values()[0] - 100.0).abs() < 1e-12);
    assert!((anc_out.values()[1] - (200.0 / 3.0)).abs() < 1e-12);
    let huge_ramp = relative_strength_index(
        &mixed,
        Some(Window::Days {
            size: 1_000_000,
            ramp: 1_000_000,
        }),
    )
    .unwrap();
    assert!(huge_ramp.dates().is_empty() && huge_ramp.values().is_empty());
}

#[test]
fn rsi_monotonic_constant_nan_inf_and_determinism() {
    let dates10: Vec<_> = (1..=10).map(|d| date(2020, 1, d)).collect();
    let dates6 = dates10[..6].to_vec();

    let inc = Series::try_new(dates10.clone(), (1..=10).map(f64::from).collect()).unwrap();
    let dec = Series::try_new(dates10.clone(), (1..=10).rev().map(f64::from).collect()).unwrap();
    let flat = Series::try_new(dates10.clone(), vec![5.0; 10]).unwrap();
    let nan_mid = Series::try_new(dates6.clone(), vec![1.0, 2.0, f64::NAN, 4.0, 3.0, 5.0]).unwrap();
    let inf_mid =
        Series::try_new(dates6.clone(), vec![1.0, 2.0, f64::INFINITY, 4.0, 3.0, 5.0]).unwrap();
    let ninf_mid = Series::try_new(
        dates6.clone(),
        vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 3.0, 5.0],
    )
    .unwrap();

    let cases = [
        (&inc, Window::Observations { size: 3, ramp: 3 }),
        (&dec, Window::Observations { size: 3, ramp: 3 }),
        (&flat, Window::Observations { size: 3, ramp: 3 }),
        (&nan_mid, Window::Observations { size: 2, ramp: 0 }),
        (&inf_mid, Window::Observations { size: 2, ramp: 0 }),
        (&ninf_mid, Window::Observations { size: 2, ramp: 0 }),
    ];
    for (series, window) in cases {
        let a = relative_strength_index(series, Some(window)).unwrap();
        let b = relative_strength_index(series, Some(window)).unwrap();
        assert_eq!(a.dates(), b.dates());
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    let r_inc =
        relative_strength_index(&inc, Some(Window::Observations { size: 3, ramp: 3 })).unwrap();
    let r_dec =
        relative_strength_index(&dec, Some(Window::Observations { size: 3, ramp: 3 })).unwrap();
    let r_flat =
        relative_strength_index(&flat, Some(Window::Observations { size: 3, ramp: 3 })).unwrap();
    assert_eq!(r_inc.values(), &[100.0; 6]);
    assert_eq!(r_dec.values(), &[0.0; 6]);
    assert_eq!(r_flat.values(), &[100.0; 6]);

    let r_nan =
        relative_strength_index(&nan_mid, Some(Window::Observations { size: 2, ramp: 0 })).unwrap();
    assert_eq!(r_nan.dates(), &dates6[1..]);
    assert_eq!(r_nan.values()[0], 100.0);
    assert!(r_nan.values()[1..].iter().all(|v| v.is_nan()));

    let r_inf =
        relative_strength_index(&inf_mid, Some(Window::Observations { size: 2, ramp: 0 })).unwrap();
    assert_eq!(r_inf.dates(), &dates6[1..]);
    assert_eq!(&r_inf.values()[..2], &[100.0, 100.0]);
    assert!(r_inf.values()[2..].iter().all(|v| v.is_nan()));

    let r_ninf =
        relative_strength_index(&ninf_mid, Some(Window::Observations { size: 2, ramp: 0 }))
            .unwrap();
    assert_eq!(r_ninf.dates(), &dates6[1..]);
    assert_eq!(&r_ninf.values()[..2], &[100.0, 0.0]);
    assert!(r_ninf.values()[2..].iter().all(|v| v.is_nan()));
}

#[test]
fn rsi_empty_short_expanding_and_error_boundaries() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for w in [
        None,
        Some(Window::Expanding),
        Some(Window::Observations { size: 3, ramp: 3 }),
        Some(Window::Observations { size: 0, ramp: 0 }),
        Some(Window::Days { size: 3, ramp: 0 }),
    ] {
        let out = relative_strength_index(&empty, w).unwrap();
        assert!(out.dates().is_empty() && out.values().is_empty());
    }
    assert_eq!(
        relative_strength_index(&empty, Some(Window::Days { size: 3, ramp: 3 })),
        Err(Error::Ramp)
    );

    // Length 1: normalized against x (len 1), then gains is empty
    let single = Series::try_new(vec![date(2020, 1, 1)], vec![5.0]).unwrap();
    for w in [
        Some(Window::Expanding),
        Some(Window::Observations { size: 1, ramp: 0 }),
        Some(Window::Observations { size: 1, ramp: 1 }),
        Some(Window::Observations { size: 2, ramp: 0 }),
        Some(Window::Days { size: 3, ramp: 0 }),
    ] {
        let out = relative_strength_index(&single, w).unwrap();
        assert!(out.dates().is_empty() && out.values().is_empty());
    }
    assert_eq!(relative_strength_index(&single, None), Err(Error::Ramp));
    assert_eq!(
        relative_strength_index(&single, Some(Window::Observations { size: 2, ramp: 2 })),
        Err(Error::Ramp)
    );
    assert_eq!(
        relative_strength_index(&single, Some(Window::Days { size: 3, ramp: 3 })),
        Err(Error::Ramp)
    );
    assert_eq!(
        relative_strength_index(&single, Some(Window::Observations { size: 0, ramp: 0 })),
        Err(Error::Window)
    );
    assert_eq!(
        relative_strength_index(&single, Some(Window::Days { size: 0, ramp: 0 })),
        Err(Error::Window)
    );

    // Length 2: gains has length 1
    let two = Series::try_new(vec![date(2020, 1, 1), date(2020, 1, 2)], vec![5.0, 6.0]).unwrap();
    let two_obs1 =
        relative_strength_index(&two, Some(Window::Observations { size: 1, ramp: 0 })).unwrap();
    assert_eq!(two_obs1.dates(), &[date(2020, 1, 2)]);
    assert_eq!(two_obs1.values(), &[100.0]);
    for w in [
        Some(Window::Expanding),
        Some(Window::Observations { size: 1, ramp: 1 }),
        Some(Window::Observations { size: 2, ramp: 0 }),
    ] {
        let out = relative_strength_index(&two, w).unwrap();
        assert!(out.dates().is_empty() && out.values().is_empty());
    }
    assert_eq!(
        relative_strength_index(&two, Some(Window::Observations { size: 2, ramp: 2 })),
        Err(Error::Ramp)
    );

    // Multi-point series: Window::Expanding -> empty; ramp > len(gains) -> Error::Ramp; size == 0 -> Error::Window
    let dates10: Vec<_> = (1..=10).map(|d| date(2020, 1, d)).collect();
    let mixed = Series::try_new(
        dates10,
        vec![10.0, 12.0, 11.0, 13.0, 13.0, 9.0, 10.5, 14.0, 12.0, 15.0],
    )
    .unwrap();
    assert!(relative_strength_index(&mixed, Some(Window::Expanding))
        .unwrap()
        .values()
        .is_empty());
    assert!(
        relative_strength_index(&mixed, Some(Window::Observations { size: 3, ramp: 9 }))
            .unwrap()
            .values()
            .is_empty()
    );
    assert_eq!(relative_strength_index(&mixed, None), Err(Error::Ramp));
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
    assert_eq!(
        relative_strength_index(&mixed, Some(Window::Days { size: 0, ramp: 3 })),
        Err(Error::Window)
    );
}

#[test]
fn macd_defaults_specified_periods_and_signal_smoothing() {
    let dates10: Vec<_> = (1..=10).map(|d| date(2020, 1, d)).collect();
    let prices10 = Series::try_new(
        dates10.clone(),
        vec![
            100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 111.0, 110.0, 115.0,
        ],
    )
    .unwrap();

    // Default (None, None, None) equals explicit (12, 26, 1) bit-for-bit
    let def = macd(&prices10, None, None, None).unwrap();
    let exp_def = macd(&prices10, Some(12), Some(26), Some(1)).unwrap();
    assert_eq!(def.dates(), dates10.as_slice());
    assert_eq!(def.dates(), exp_def.dates());
    assert!(def
        .values()
        .iter()
        .zip(exp_def.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(def.values()[0] == 0.0 && def.values()[0].is_sign_positive());
    assert!((def.values()[1] - 0.15954415954415424).abs() < 1e-12);
    assert!((def.values()[9] - 2.881299976008549).abs() < 1e-12);

    // Signal-line smoothing s = 9
    let sig9 = macd(&prices10, Some(12), Some(26), Some(9)).unwrap();
    assert_eq!(sig9.dates(), dates10.as_slice());
    assert!(sig9.values()[0] == 0.0 && sig9.values()[0].is_sign_positive());
    assert!((sig9.values()[1] - 0.03190883190883085).abs() < 1e-12);
    assert!((sig9.values()[9] - 1.533517851885833).abs() < 1e-12);

    // Upstream test_macd assertions: (10, 10), (10, 12, s=1) == (10, 12), and (2, 3)
    let dates6: Vec<_> = (1..=6).map(|d| date(2019, 1, d)).collect();
    let sample = Series::try_new(dates6.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    let eq_10 = macd(&sample, Some(10), Some(10), None).unwrap();
    assert_eq!(eq_10.dates(), dates6.as_slice());
    assert!(eq_10
        .values()
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));

    let m10_12_def = macd(&sample, Some(10), Some(12), None).unwrap();
    let m10_12_s1 = macd(&sample, Some(10), Some(12), Some(1)).unwrap();
    assert!(m10_12_def
        .values()
        .iter()
        .zip(m10_12_s1.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!((m10_12_s1.values()[5] - 0.05817814161076118).abs() < 1e-12);

    let m2_3 = macd(&sample, Some(2), Some(3), None).unwrap();
    let expected_2_3 = [
        0.0,
        -0.16666666666666696,
        0.02777777777777768,
        -0.28240740740740744,
        0.09336419753086433,
        0.6248713991769552,
    ];
    for (&actual, &expected) in m2_3.values().iter().zip(&expected_2_3) {
        assert!((actual - expected).abs() < 1e-12);
    }

    // Signal smoothing s = 4 on (2, 3)
    let m2_3_4 = macd(&sample, Some(2), Some(3), Some(4)).unwrap();
    assert!((m2_3_4.values()[1] - (-0.06666666666666679)).abs() < 1e-12);
    assert!((m2_3_4.values()[5] - 0.22544930041152286).abs() < 1e-12);
}

#[test]
fn macd_span_combinations_and_irregular_leap_dates() {
    let dates6: Vec<_> = (1..=6).map(|d| date(2019, 1, d)).collect();
    let sample = Series::try_new(dates6.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // m == n (including all 1s) yields +0.0 everywhere
    let all_1 = macd(&sample, Some(1), Some(1), Some(1)).unwrap();
    assert!(all_1
        .values()
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));

    // m > n negates (n, m)
    let fwd = macd(&sample, Some(2), Some(5), Some(3)).unwrap();
    let inv = macd(&sample, Some(5), Some(2), Some(3)).unwrap();
    assert!((inv.values()[1] - 0.16666666666666674).abs() < 1e-12);
    assert!((inv.values()[5] - (-0.5369084362139918)).abs() < 1e-12);
    for (&f, &r) in fwd.values().iter().zip(inv.values()) {
        assert!((f + r).abs() < 1e-12);
    }

    // m = 1, n = 3, s = 2
    let m1_n3_s2 = macd(&sample, Some(1), Some(3), Some(2)).unwrap();
    assert!((m1_n3_s2.values()[1] - (-1.0 / 3.0)).abs() < 1e-12);
    assert!((m1_n3_s2.values()[5] - 1.2497427983539096).abs() < 1e-12);

    // Irregular dates including leap day 2020-02-29: observation-order EWM matches consecutive dates
    let leap_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 2, 29),
        date(2020, 3, 2),
        date(2020, 3, 10),
    ];
    let reg_dates: Vec<_> = (1..=5).map(|d| date(2020, 1, d)).collect();
    let vals = vec![10.0, 12.0, 11.0, 15.0, 13.0];
    let irreg = Series::try_new(leap_dates.clone(), vals.clone()).unwrap();
    let reg = Series::try_new(reg_dates, vals).unwrap();
    let r_irreg = macd(&irreg, Some(2), Some(4), Some(3)).unwrap();
    let r_reg = macd(&reg, Some(2), Some(4), Some(3)).unwrap();
    assert_eq!(r_irreg.dates(), leap_dates.as_slice());
    assert!(r_irreg
        .values()
        .iter()
        .zip(r_reg.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!((r_irreg.values()[4] - 0.6150320987654324).abs() < 1e-12);
}

#[test]
fn macd_missing_signed_zero_overflow_and_determinism() {
    let d5: Vec<_> = (1..=5).map(|d| date(2020, 1, d)).collect();
    let d6: Vec<_> = (1..=6).map(|d| date(2020, 1, d)).collect();
    let d7: Vec<_> = (1..=7).map(|d| date(2020, 1, d)).collect();

    // Leading NaN with finite tail: subtract(STEP) backfills leading rows to +0.0 entering 3rd EWM
    let lead_nan =
        Series::try_new(d6.clone(), vec![f64::NAN, f64::NAN, 10.0, 20.0, 15.0, 25.0]).unwrap();
    let ln_s1 = macd(&lead_nan, Some(2), Some(4), Some(1)).unwrap();
    let ln_s3 = macd(&lead_nan, Some(2), Some(4), Some(3)).unwrap();
    assert!(ln_s1.values()[..3]
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));
    assert!((ln_s1.values()[3] - 2.6666666666666643).abs() < 1e-12);
    assert!(ln_s3.values()[..3]
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));
    // Smoothed against the backfilled +0.0 (2.6666666666666643 * 0.5 = 1.3333333333333321)
    assert!((ln_s3.values()[3] - 1.3333333333333321).abs() < 1e-12);
    assert!((ln_s3.values()[5] - 2.228148148148147).abs() < 1e-12);

    // Leading ±Inf behaves identically to leading NaN
    let lead_inf = Series::try_new(
        d5.clone(),
        vec![f64::INFINITY, f64::NEG_INFINITY, 10.0, 20.0, 15.0],
    )
    .unwrap();
    let li_s2 = macd(&lead_inf, Some(2), Some(4), Some(2)).unwrap();
    assert!(li_s2.values()[..3]
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));
    assert!((li_s2.values()[3] - 1.7777777777777761).abs() < 1e-12);

    // Internal and trailing NaN / ±Inf
    let internal_nan = Series::try_new(
        d7,
        vec![10.0, f64::NAN, f64::NAN, 20.0, f64::NAN, 5.0, 12.0],
    )
    .unwrap();
    let in_s3 = macd(&internal_nan, Some(2), Some(5), Some(3)).unwrap();
    assert!((in_s3.values()[3] - 2.089783281733747).abs() < 1e-12);
    assert!((in_s3.values()[6] - (-0.619508329647647)).abs() < 1e-12);

    let trailing_nan = Series::try_new(
        d6.clone(),
        vec![10.0, 20.0, 15.0, f64::NAN, f64::NAN, f64::NAN],
    )
    .unwrap();
    let tr_s3 = macd(&trailing_nan, Some(2), Some(4), Some(3)).unwrap();
    assert!((tr_s3.values()[5] - 1.166666666666666).abs() < 1e-12);

    // All missing (all NaN, all ±Inf, mixed) -> all NaN
    let all_nan = Series::try_new(d5[..4].to_vec(), vec![f64::NAN; 4]).unwrap();
    let all_inf = Series::try_new(
        d5[..3].to_vec(),
        vec![f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY],
    )
    .unwrap();
    for s in [&all_nan, &all_inf] {
        let out = macd(s, Some(2), Some(3), Some(2)).unwrap();
        assert_eq!(out.dates(), s.dates());
        assert!(out.values().iter().all(|v| v.is_nan()));
    }

    // Signed zero: constant -0.0, mixed signed zeros, and leading NaN + -0.0 all emit +0.0
    let neg_zero = Series::try_new(d5.clone(), vec![-0.0; 5]).unwrap();
    let mixed_zero = Series::try_new(d5.clone(), vec![-0.0, 0.0, -0.0, 0.0, -0.0]).unwrap();
    for s in [&neg_zero, &mixed_zero] {
        let out = macd(s, Some(2), Some(5), Some(3)).unwrap();
        assert!(out
            .values()
            .iter()
            .all(|v| *v == 0.0 && v.is_sign_positive()));
    }

    // Large sign-flip without overflow and true a - b overflow to ±Inf entering 3rd EWM
    let large_flip =
        Series::try_new(d5.clone(), vec![1e308, -1e308, 1e308, -1e308, 1e308]).unwrap();
    let true_ovf =
        Series::try_new(d5, vec![1.7e308, -1.7e308, 1.7e308, -1.7e308, 1.7e308]).unwrap();
    let ovf_out = macd(&true_ovf, Some(1), Some(20), Some(3)).unwrap();
    assert!(ovf_out.values()[0] == 0.0 && ovf_out.values()[0].is_sign_positive());
    assert!(ovf_out.values()[1] == 0.0 && ovf_out.values()[1].is_sign_positive());
    assert!((ovf_out.values()[2] - 1.9531368102796675e307).abs() <= 1e-10 * 1.9531368102796675e307);
    assert_eq!(ovf_out.values()[3].to_bits(), ovf_out.values()[2].to_bits());
    assert!((ovf_out.values()[4] - 4.2030086915088527e307).abs() <= 1e-10 * 4.2030086915088527e307);

    // Back-to-back determinism across all edge cases
    let edge_cases = [
        (&lead_nan, Some(2), Some(4), Some(3)),
        (&lead_inf, Some(2), Some(4), Some(2)),
        (&internal_nan, Some(2), Some(5), Some(3)),
        (&trailing_nan, Some(2), Some(4), Some(3)),
        (&all_nan, Some(2), Some(3), Some(2)),
        (&all_inf, Some(2), Some(3), Some(2)),
        (&neg_zero, Some(2), Some(5), Some(3)),
        (&large_flip, Some(2), Some(10), Some(2)),
        (&true_ovf, Some(1), Some(20), Some(3)),
    ];
    for (series, m, n, s) in edge_cases {
        let a = macd(series, m, n, s).unwrap();
        let b = macd(series, m, n, s).unwrap();
        assert_eq!(a.dates(), b.dates());
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

#[test]
fn macd_empty_singleton_and_span_errors() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    let sample =
        Series::try_new(vec![date(2020, 1, 1), date(2020, 1, 2)], vec![10.0, 12.0]).unwrap();

    // Zero spans return Error::Span on both empty and non-empty inputs
    for series in [&empty, &sample] {
        assert_eq!(macd(series, Some(0), Some(26), Some(1)), Err(Error::Span));
        assert_eq!(macd(series, Some(12), Some(0), Some(1)), Err(Error::Span));
        assert_eq!(macd(series, Some(12), Some(26), Some(0)), Err(Error::Span));
        assert_eq!(macd(series, Some(0), None, None), Err(Error::Span));
        assert_eq!(macd(series, None, Some(0), None), Err(Error::Span));
        assert_eq!(macd(series, None, None, Some(0)), Err(Error::Span));
    }

    // Valid spans on empty input return empty Series
    for (m, n, s) in [
        (None, None, None),
        (Some(1), Some(1), Some(1)),
        (Some(2), Some(3), Some(4)),
    ] {
        let out = macd(&empty, m, n, s).unwrap();
        assert!(out.dates().is_empty() && out.values().is_empty());
    }

    // Length-1 inputs: finite / -0.0 -> [+0.0]; NaN / ±Inf -> [NaN]
    let d1 = vec![date(2020, 1, 1)];
    for v in [42.0, -0.0] {
        let s = Series::try_new(d1.clone(), vec![v]).unwrap();
        let out = macd(&s, Some(12), Some(26), Some(9)).unwrap();
        assert_eq!(out.dates(), d1.as_slice());
        assert!(out.values()[0] == 0.0 && out.values()[0].is_sign_positive());
    }
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let s = Series::try_new(d1.clone(), vec![v]).unwrap();
        let out = macd(&s, Some(2), Some(3), Some(2)).unwrap();
        assert_eq!(out.dates(), d1.as_slice());
        assert!(out.values()[0].is_nan());
    }
}

#[test]
fn prices_upstream_test_prices_and_defaults() {
    let dates: Vec<_> = (1..=6).map(|d| date(2019, 1, d)).collect();
    let simple_r =
        Series::try_new(dates.clone(), vec![f64::NAN, 0.01, 0.02, -0.02, 0.0, 0.02]).unwrap();

    // Default initial (1.0) and default kind (Simple)
    let def = prices(&simple_r, None, None).unwrap();
    let expected_def = [1.0, 1.01, 1.0302, 1.009596, 1.009596, 1.02978792];
    assert_eq!(def.dates(), dates.as_slice());
    for (&a, &e) in def.values().iter().zip(&expected_def) {
        assert!((a - e).abs() <= 1e-12 + 1e-10 * e.abs(), "{a} vs {e}");
    }

    // Explicit initial=100.0 with default kind vs explicit Simple
    let init100_def = prices(&simple_r, Some(100.0), None).unwrap();
    let init100_exp = prices(&simple_r, Some(100.0), Some(ReturnKind::Simple)).unwrap();
    let expected_100 = [100.0, 101.0, 103.02, 100.9596, 100.9596, 102.978792];
    assert_eq!(init100_def.dates(), dates.as_slice());
    assert_eq!(init100_exp.dates(), dates.as_slice());
    for (i, &e) in expected_100.iter().enumerate() {
        assert_eq!(
            init100_def.values()[i].to_bits(),
            init100_exp.values()[i].to_bits()
        );
        assert!((init100_exp.values()[i] - e).abs() <= 1e-12 + 1e-10 * e.abs());
    }

    // Logarithmic and Absolute modes from upstream test_prices
    let log_r = Series::try_new(
        dates.clone(),
        vec![f64::NAN, 0.009950, 0.019803, -0.020203, 0.0, 0.019803],
    )
    .unwrap();
    let log_p = prices(&log_r, Some(100.0), Some(ReturnKind::Logarithmic)).unwrap();
    assert_eq!(log_p.dates(), dates.as_slice());
    for (&a, &e) in log_p.values().iter().zip(&expected_100) {
        assert!((a - e).abs() < 1e-3, "{a} vs {e}");
    }

    let abs_r = Series::try_new(
        dates.clone(),
        vec![f64::NAN, 1.0, 2.02, -2.0604, 0.0, 2.019192],
    )
    .unwrap();
    let abs_p = prices(&abs_r, Some(100.0), Some(ReturnKind::Absolute)).unwrap();
    assert_eq!(abs_p.dates(), dates.as_slice());
    for (&a, &e) in abs_p.values().iter().zip(&expected_100) {
        assert!((a - e).abs() <= 1e-12 + 1e-10 * e.abs(), "{a} vs {e}");
    }
}

#[test]
fn prices_roundtrip_and_leap_dates() {
    let leap_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 2, 29),
        date(2020, 3, 2),
        date(2020, 3, 10),
    ];
    let px = Series::try_new(leap_dates.clone(), vec![100.0, 105.0, 102.9, 108.0, 99.5]).unwrap();

    for kind in [
        ReturnKind::Simple,
        ReturnKind::Logarithmic,
        ReturnKind::Absolute,
    ] {
        let r = returns(&px, 1, kind).unwrap();
        let restored = prices(&r, Some(px.values()[0]), Some(kind)).unwrap();
        assert_eq!(restored.dates(), leap_dates.as_slice());
        for (&actual, &expected) in restored.values().iter().zip(px.values()) {
            assert!(
                (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                "{kind:?}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn prices_missing_infinities_collapse_and_signed_zeros() {
    let d5: Vec<_> = (1..=5).map(|d| date(2020, 1, d)).collect();
    let d4 = d5[..4].to_vec();
    let d3 = d5[..3].to_vec();
    let d2 = d5[..2].to_vec();

    // Leading, internal, and all-missing NaNs
    let lead_nan =
        Series::try_new(d5.clone(), vec![f64::NAN, f64::NAN, 0.1, f64::NAN, -0.2]).unwrap();
    let lead_s = prices(&lead_nan, Some(100.0), Some(ReturnKind::Simple)).unwrap();
    assert_eq!(lead_s.dates(), d5.as_slice());
    assert_eq!(lead_s.values()[0], 100.0);
    assert_eq!(lead_s.values()[1], 100.0);
    assert!((lead_s.values()[2] - 110.0).abs() < 1e-12);
    assert!((lead_s.values()[3] - 110.0).abs() < 1e-12);
    assert!((lead_s.values()[4] - 88.0).abs() < 1e-12);

    let all_nan = Series::try_new(d4.clone(), vec![f64::NAN; 4]).unwrap();
    for kind in [
        ReturnKind::Simple,
        ReturnKind::Logarithmic,
        ReturnKind::Absolute,
    ] {
        let out = prices(&all_nan, Some(10.0), Some(kind)).unwrap();
        assert_eq!(out.dates(), d4.as_slice());
        assert_eq!(out.values(), &[10.0; 4]);
    }

    // -1.0 collapse and -0.0 sign flip on subsequent negative factor (1 + (-2.0) = -1.0)
    let collapse = Series::try_new(d4.clone(), vec![0.1, -1.0, 0.5, -2.0]).unwrap();
    let col_out = prices(&collapse, Some(100.0), Some(ReturnKind::Simple)).unwrap();
    assert!((col_out.values()[0] - 110.0).abs() < 1e-12);
    assert!(col_out.values()[1] == 0.0 && col_out.values()[1].is_sign_positive());
    assert!(col_out.values()[2] == 0.0 && col_out.values()[2].is_sign_positive());
    assert!(col_out.values()[3] == 0.0 && col_out.values()[3].is_sign_negative());

    // ±Inf skipping across Simple, Logarithmic (+Inf skipped, -Inf -> exp(-Inf) = 0.0), Absolute
    let s_inf_pos = Series::try_new(d4.clone(), vec![0.1, f64::INFINITY, 0.2, -1.0]).unwrap();
    let s_inf_out = prices(&s_inf_pos, Some(10.0), Some(ReturnKind::Simple)).unwrap();
    assert!((s_inf_out.values()[1] - 11.0).abs() < 1e-12);
    assert!((s_inf_out.values()[2] - 13.2).abs() < 1e-12);
    assert!(s_inf_out.values()[3] == 0.0 && s_inf_out.values()[3].is_sign_positive());

    let log_inf =
        Series::try_new(d4.clone(), vec![0.1, f64::INFINITY, 0.2, f64::NEG_INFINITY]).unwrap();
    let log_inf_out = prices(&log_inf, Some(10.0), Some(ReturnKind::Logarithmic)).unwrap();
    assert!((log_inf_out.values()[1] - 10.0 * 0.1_f64.exp()).abs() < 1e-12);
    assert!((log_inf_out.values()[2] - 10.0 * 0.1_f64.exp() * 0.2_f64.exp()).abs() < 1e-12);
    assert!(log_inf_out.values()[3] == 0.0 && log_inf_out.values()[3].is_sign_positive());

    let abs_inf = Series::try_new(d4, vec![1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY]).unwrap();
    let abs_inf_out = prices(&abs_inf, Some(10.0), Some(ReturnKind::Absolute)).unwrap();
    assert_eq!(abs_inf_out.values(), &[11.0, 11.0, 13.0, 13.0]);

    // initial boundaries: +0.0, -0.0, negative, NaN, +Inf, -Inf
    let sample3 = Series::try_new(d3.clone(), vec![0.1, -0.2, 0.5]).unwrap();
    let z_pos = prices(&sample3, Some(0.0), Some(ReturnKind::Simple)).unwrap();
    assert!(z_pos
        .values()
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_positive()));
    let z_neg = prices(&sample3, Some(-0.0), Some(ReturnKind::Simple)).unwrap();
    assert!(z_neg
        .values()
        .iter()
        .all(|v| *v == 0.0 && v.is_sign_negative()));
    let init_neg = prices(&sample3, Some(-10.0), Some(ReturnKind::Simple)).unwrap();
    assert!((init_neg.values()[0] - (-11.0)).abs() < 1e-12);
    assert!((init_neg.values()[1] - (-8.8)).abs() < 1e-12);
    assert!((init_neg.values()[2] - (-13.2)).abs() < 1e-12);
    let init_nan = prices(&sample3, Some(f64::NAN), Some(ReturnKind::Simple)).unwrap();
    assert!(init_nan.values().iter().all(|v| v.is_nan()));
    let collapse3 = Series::try_new(d3.clone(), vec![0.1, -1.0, 0.2]).unwrap();
    let init_inf = prices(&collapse3, Some(f64::INFINITY), Some(ReturnKind::Simple)).unwrap();
    assert!(init_inf.values()[0].is_infinite() && init_inf.values()[0].is_sign_positive());
    assert!(init_inf.values()[1].is_nan() && init_inf.values()[2].is_nan());

    // Absolute signed zero: empty prefix emits +0.0 + (-0.0) = +0.0; finite -0.0 preserves -0.0
    let abs_nz = Series::try_new(d3.clone(), vec![-0.0, -0.0, 0.0]).unwrap();
    let abs_nz_out = prices(&abs_nz, Some(-0.0), Some(ReturnKind::Absolute)).unwrap();
    assert!(abs_nz_out.values()[0] == 0.0 && abs_nz_out.values()[0].is_sign_negative());
    assert!(abs_nz_out.values()[1] == 0.0 && abs_nz_out.values()[1].is_sign_negative());
    assert!(abs_nz_out.values()[2] == 0.0 && abs_nz_out.values()[2].is_sign_positive());

    let abs_lead_nan_nz = Series::try_new(d3.clone(), vec![f64::NAN, -0.0, 0.0]).unwrap();
    let abs_lead_out = prices(&abs_lead_nan_nz, Some(-0.0), Some(ReturnKind::Absolute)).unwrap();
    assert!(abs_lead_out.values()[0] == 0.0 && abs_lead_out.values()[0].is_sign_positive());
    assert!(abs_lead_out.values()[1] == 0.0 && abs_lead_out.values()[1].is_sign_negative());
    assert!(abs_lead_out.values()[2] == 0.0 && abs_lead_out.values()[2].is_sign_positive());

    // Log underflow (-800.0 -> 0.0) and cumulative product overflow ([709.0, 709.0] -> +Inf)
    let log_under = Series::try_new(d3, vec![0.1, -800.0, 0.2]).unwrap();
    let log_under_out = prices(&log_under, Some(10.0), Some(ReturnKind::Logarithmic)).unwrap();
    assert!((log_under_out.values()[0] - 10.0 * 0.1_f64.exp()).abs() < 1e-12);
    assert!(log_under_out.values()[1] == 0.0 && log_under_out.values()[1].is_sign_positive());
    assert!(log_under_out.values()[2] == 0.0 && log_under_out.values()[2].is_sign_positive());

    let log_cumprod = Series::try_new(d2, vec![709.0, 709.0]).unwrap();
    let log_cumprod_out = prices(&log_cumprod, Some(1.0), Some(ReturnKind::Logarithmic)).unwrap();
    assert!(log_cumprod_out.values()[0].is_finite());
    assert!(
        log_cumprod_out.values()[1].is_infinite() && log_cumprod_out.values()[1].is_sign_positive()
    );

    let simple_cumprod =
        Series::try_new(d5, vec![1e200, 1e200, f64::NAN, f64::INFINITY, -1.0]).unwrap();
    let sc_out = prices(&simple_cumprod, Some(1.0), Some(ReturnKind::Simple)).unwrap();
    assert!(sc_out.values()[1].is_infinite() && sc_out.values()[1].is_sign_positive());
    assert!(sc_out.values()[2].is_infinite() && sc_out.values()[3].is_infinite());
    assert!(sc_out.values()[4].is_nan());

    // Back-to-back determinism across all edge cases
    let edge_cases = [
        (&lead_nan, Some(100.0), Some(ReturnKind::Simple)),
        (&all_nan, Some(10.0), Some(ReturnKind::Logarithmic)),
        (&collapse, Some(100.0), Some(ReturnKind::Simple)),
        (&s_inf_pos, Some(10.0), Some(ReturnKind::Simple)),
        (&log_inf, Some(10.0), Some(ReturnKind::Logarithmic)),
        (&abs_inf, Some(10.0), Some(ReturnKind::Absolute)),
        (&sample3, Some(-0.0), Some(ReturnKind::Simple)),
        (&sample3, Some(f64::NAN), Some(ReturnKind::Simple)),
        (&collapse3, Some(f64::INFINITY), Some(ReturnKind::Simple)),
        (&abs_nz, Some(-0.0), Some(ReturnKind::Absolute)),
        (&abs_lead_nan_nz, Some(-0.0), Some(ReturnKind::Absolute)),
        (&log_under, Some(10.0), Some(ReturnKind::Logarithmic)),
        (&log_cumprod, Some(1.0), Some(ReturnKind::Logarithmic)),
        (&simple_cumprod, Some(1.0), Some(ReturnKind::Simple)),
    ];
    for (series, initial, kind) in edge_cases {
        let a = prices(series, initial, kind).unwrap();
        let b = prices(series, initial, kind).unwrap();
        assert_eq!(a.dates(), b.dates());
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

#[test]
fn prices_empty_and_logarithmic_overflow_error() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for kind in [
        None,
        Some(ReturnKind::Simple),
        Some(ReturnKind::Logarithmic),
        Some(ReturnKind::Absolute),
    ] {
        for initial in [
            None,
            Some(100.0),
            Some(-0.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ] {
            let out = prices(&empty, initial, kind).unwrap();
            assert!(out.dates().is_empty() && out.values().is_empty());
        }
    }

    let d3: Vec<_> = (1..=3).map(|d| date(2020, 1, d)).collect();
    let ovf_mid = Series::try_new(d3, vec![0.1, 800.0, 0.2]).unwrap();
    assert_eq!(
        prices(&ovf_mid, Some(10.0), Some(ReturnKind::Logarithmic)),
        Err(Error::Overflow)
    );
    // Overflow check runs before multiplying by initial (even NaN) and even after +Inf
    let ovf_after_inf = Series::try_new(
        vec![date(2020, 1, 1), date(2020, 1, 2)],
        vec![f64::INFINITY, 800.0],
    )
    .unwrap();
    assert_eq!(
        prices(
            &ovf_after_inf,
            Some(f64::NAN),
            Some(ReturnKind::Logarithmic)
        ),
        Err(Error::Overflow)
    );
    // Tight boundary around f64::MAX.ln() (~709.782712893384)
    let near_max = Series::try_new(vec![date(2020, 1, 1)], vec![709.782_712_893_384]).unwrap();
    assert!(prices(&near_max, Some(1.0), Some(ReturnKind::Logarithmic))
        .unwrap()
        .values()[0]
        .is_finite());
    let just_over = Series::try_new(vec![date(2020, 1, 1)], vec![709.7827128933841]).unwrap();
    assert_eq!(
        prices(&just_over, Some(1.0), Some(ReturnKind::Logarithmic)),
        Err(Error::Overflow)
    );
}

#[test]
fn std_upstream_test_std_and_windows() {
    let dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 7),
        date(2019, 1, 8),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // 1. Expanding (upstream default Window(None, 0))
    let exp = series_std(&x, Window::Expanding).unwrap();
    assert_eq!(exp.dates(), dates.as_slice());
    assert!(exp.values()[0].is_nan());
    let expected_exp = [
        std::f64::consts::FRAC_1_SQRT_2,
        0.5773502691896258,
        0.9574271077563381,
        0.8944271909999159,
        1.6733200530681511,
    ];
    for (&a, &e) in exp.values()[1..].iter().zip(&expected_exp) {
        assert!((a - e).abs() <= 1e-12 + 1e-10 * e.abs(), "{a} vs {e}");
    }

    // 2. Observation window Window(2, 0)
    let obs2 = series_std(&x, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(obs2.dates(), dates.as_slice());
    assert!(obs2.values()[0].is_nan());
    let expected_obs2 = [
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::SQRT_2,
        std::f64::consts::SQRT_2,
        2.1213203435596424,
    ];
    for (&a, &e) in obs2.values()[1..].iter().zip(&expected_obs2) {
        assert!((a - e).abs() <= 1e-12 + 1e-10 * e.abs(), "{a} vs {e}");
    }

    // 3. Calendar-day window 7d (upstream Window('1w', 0))
    let cal7d = series_std(&x, Window::Days { size: 7, ramp: 0 }).unwrap();
    assert_eq!(cal7d.dates(), dates.as_slice());
    assert!(cal7d.values()[0].is_nan());
    let expected_cal7d = [
        std::f64::consts::FRAC_1_SQRT_2,
        0.5773502691896258,
        0.9574271077563381,
        0.8944271909999159,
        1.8708286933869707,
    ];
    for (&a, &e) in cal7d.values()[1..].iter().zip(&expected_cal7d) {
        assert!((a - e).abs() <= 1e-12 + 1e-10 * e.abs(), "{a} vs {e}");
    }

    // 4. Observation window with ramp: Window(3, 2)
    let obs3_r2 = series_std(&x, Window::Observations { size: 3, ramp: 2 }).unwrap();
    assert_eq!(obs3_r2.dates(), &dates[2..]);
    let expected_obs3_r2 = [
        0.5773502691896258,
        1.0,
        1.1547005383792515,
        2.5166114784235836,
    ];
    for (&a, &e) in obs3_r2.values().iter().zip(&expected_obs3_r2) {
        assert!((a - e).abs() <= 1e-12 + 1e-10 * e.abs(), "{a} vs {e}");
    }
}

#[test]
fn std_leap_day_irregular_calendar_windows() {
    let leap_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 2, 29),
        date(2020, 3, 2),
        date(2020, 3, 5),
    ];
    let leap = Series::try_new(leap_dates.clone(), vec![1.0, 3.0, 2.0, 6.0, 10.0]).unwrap();

    // Window::Days { size: 3, ramp: 3 }: half-open (date - 3d, date], keeps Mar 2 and Mar 5
    let d3_r3 = series_std(&leap, Window::Days { size: 3, ramp: 3 }).unwrap();
    assert_eq!(d3_r3.dates(), &[date(2020, 3, 2), date(2020, 3, 5)]);
    assert!((d3_r3.values()[0] - 2.8284271247461903).abs() < 1e-12);
    assert!(d3_r3.values()[1].is_nan());

    // Window::Days { size: 3, ramp: 0 }
    let d3_r0 = series_std(&leap, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert_eq!(d3_r0.dates(), leap_dates.as_slice());
    assert!(d3_r0.values()[0].is_nan());
    assert!((d3_r0.values()[1] - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((d3_r0.values()[2] - 1.0).abs() < 1e-12);
    assert!((d3_r0.values()[3] - 2.8284271247461903).abs() < 1e-12);
    assert!(d3_r0.values()[4].is_nan());

    // Window::Days { size: 7, ramp: 0 } crossing leap day:
    // Mar 5 - 7d = Feb 27 (excluded because half-open > Feb 27), so [3, 2, 6, 10]
    let d7_r0 = series_std(&leap, Window::Days { size: 7, ramp: 0 }).unwrap();
    assert_eq!(d7_r0.dates(), leap_dates.as_slice());
    assert!(d7_r0.values()[0].is_nan());
    assert!((d7_r0.values()[3] - 2.160246899469287).abs() < 1e-12);
    assert!((d7_r0.values()[4] - 3.593976442141304).abs() < 1e-12);

    // Window::Days { size: 1, ramp: 0 } -> all NaN; oversize ramp -> empty
    let d1 = series_std(&leap, Window::Days { size: 1, ramp: 0 }).unwrap();
    assert_eq!(d1.dates(), leap_dates.as_slice());
    assert!(d1.values().iter().all(|v| v.is_nan()));
    let d30 = series_std(&leap, Window::Days { size: 30, ramp: 30 }).unwrap();
    assert!(d30.dates().is_empty() && d30.values().is_empty());
}

#[test]
fn std_constants_missing_infinities_overflow_and_determinism() {
    let dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 7),
        date(2019, 1, 8),
    ];

    // Constant series (4.2, 0.1, -0.0, mixed signed zeros) emit exact +0.0 when count >= 2
    let c42 = Series::try_new(dates.clone(), vec![4.2; 6]).unwrap();
    let c01 = Series::try_new(dates.clone(), vec![0.1; 6]).unwrap();
    let cneg0 = Series::try_new(dates.clone(), vec![-0.0; 6]).unwrap();
    let cmix0 = Series::try_new(dates.clone(), vec![-0.0, 0.0, -0.0, 0.0, -0.0, 0.0]).unwrap();
    for s in [&c42, &c01, &cneg0, &cmix0] {
        let obs = series_std(s, Window::Observations { size: 3, ramp: 0 }).unwrap();
        assert!(obs.values()[0].is_nan());
        assert!(obs.values()[1..]
            .iter()
            .all(|v| *v == 0.0 && v.is_sign_positive()));
        let cal = series_std(s, Window::Days { size: 3, ramp: 0 }).unwrap();
        assert!(cal.values()[0].is_nan());
        assert!(cal.values()[4].is_nan());
        for idx in [1, 2, 3, 5] {
            assert!(cal.values()[idx] == 0.0 && cal.values()[idx].is_sign_positive());
        }
    }

    // Mid-series NaN and all-NaN
    let mid_nan = Series::try_new(
        dates.clone(),
        vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, 5.0],
    )
    .unwrap();
    let nan_obs = series_std(&mid_nan, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!(nan_obs.values()[0].is_nan() && nan_obs.values()[1].is_nan());
    assert!((nan_obs.values()[2] - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!(nan_obs.values()[3..].iter().all(|v| v.is_nan()));
    let nan_days = series_std(&mid_nan, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert!(nan_days.values()[0].is_nan() && nan_days.values()[1].is_nan());
    assert!((nan_days.values()[2] - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((nan_days.values()[3] - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!(nan_days.values()[4].is_nan() && nan_days.values()[5].is_nan());

    let all_nan = Series::try_new(dates.clone(), vec![f64::NAN; 6]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        let out = series_std(&all_nan, w).unwrap();
        assert_eq!(out.dates(), dates.as_slice());
        assert!(out.values().iter().all(|v| v.is_nan()));
    }

    // ±Inf: skipped in Expanding / Observations vs NaN in Days
    let infs = Series::try_new(
        dates.clone(),
        vec![1.0, f64::INFINITY, 3.0, 5.0, f64::NEG_INFINITY, 7.0],
    )
    .unwrap();
    let inf_obs = series_std(&infs, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!(inf_obs.values()[0].is_nan() && inf_obs.values()[1].is_nan());
    for &v in &inf_obs.values()[2..] {
        assert!((v - std::f64::consts::SQRT_2).abs() < 1e-12);
    }
    let inf_exp = series_std(&infs, Window::Expanding).unwrap();
    assert!(inf_exp.values()[0].is_nan() && inf_exp.values()[1].is_nan());
    assert!((inf_exp.values()[2] - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((inf_exp.values()[3] - 2.0).abs() < 1e-12);
    assert!((inf_exp.values()[4] - 2.0).abs() < 1e-12);
    assert!((inf_exp.values()[5] - 2.581988897471611).abs() < 1e-12);
    let inf_days = series_std(&infs, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert!(inf_days.values().iter().all(|v| v.is_nan()));

    // Large variance overflow (1e200): per-window two-pass produces +Inf cleanly (unlike pandas
    // rolling Welford which turns subsequent windows NaN after overflow)
    let large = Series::try_new(dates, vec![1e200, 1e200, -1e200, 1e200, -1e200, 1e200]).unwrap();
    for w in [
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 4, ramp: 0 },
    ] {
        let out = series_std(&large, w).unwrap();
        assert!(out.values()[0].is_nan());
        assert!(out.values()[1] == 0.0 && out.values()[1].is_sign_positive());
        assert!(out.values()[2..]
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive()));
    }

    // Back-to-back determinism across all edge inputs
    let checks: [(&Series, Window); 10] = [
        (&c42, Window::Observations { size: 3, ramp: 0 }),
        (&c01, Window::Days { size: 3, ramp: 0 }),
        (&cneg0, Window::Observations { size: 3, ramp: 0 }),
        (&mid_nan, Window::Observations { size: 3, ramp: 0 }),
        (&mid_nan, Window::Days { size: 4, ramp: 0 }),
        (&all_nan, Window::Observations { size: 3, ramp: 0 }),
        (&infs, Window::Observations { size: 3, ramp: 0 }),
        (&infs, Window::Days { size: 4, ramp: 0 }),
        (&large, Window::Observations { size: 3, ramp: 0 }),
        (&large, Window::Days { size: 4, ramp: 0 }),
    ];
    for (series, window) in checks {
        let a = series_std(series, window).unwrap();
        let b = series_std(series, window).unwrap();
        assert_eq!(a.dates(), b.dates());
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

#[test]
fn std_empty_singleton_size1_oversize_ramp_and_errors() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 2, ramp: 0 },
        Window::Observations { size: 0, ramp: 0 },
        Window::Observations { size: 2, ramp: 5 },
        Window::Days { size: 0, ramp: 0 },
        Window::Days { size: 2, ramp: 2 },
    ] {
        let out = series_std(&empty, w).unwrap();
        assert!(out.dates().is_empty() && out.values().is_empty());
    }

    let single = Series::try_new(vec![date(2020, 1, 1)], vec![5.0]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 1, ramp: 0 },
        Window::Days { size: 2, ramp: 0 },
    ] {
        let out = series_std(&single, w).unwrap();
        assert_eq!(out.dates(), single.dates());
        assert!(out.values()[0].is_nan());
    }
    assert!(
        series_std(&single, Window::Observations { size: 2, ramp: 0 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert!(
        series_std(&single, Window::Observations { size: 1, ramp: 1 })
            .unwrap()
            .values()
            .is_empty()
    );
    assert_eq!(
        series_std(&single, Window::Observations { size: 1, ramp: 2 }),
        Err(Error::Ramp)
    );

    let dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 7),
        date(2019, 1, 8),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // Size-1 observation window -> all NaN
    let obs1 = series_std(&x, Window::Observations { size: 1, ramp: 0 }).unwrap();
    assert_eq!(obs1.dates(), dates.as_slice());
    assert!(obs1.values().iter().all(|v| v.is_nan()));

    // Oversize observation window & ramp == len -> empty
    assert!(series_std(&x, Window::Observations { size: 10, ramp: 0 })
        .unwrap()
        .values()
        .is_empty());
    assert!(series_std(&x, Window::Observations { size: 2, ramp: 6 })
        .unwrap()
        .values()
        .is_empty());

    // Error::Window and Error::Ramp (including documented Days { size: 0 } rejection and precedence)
    assert_eq!(
        series_std(&x, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        series_std(&x, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        series_std(&x, Window::Observations { size: 0, ramp: 10 }),
        Err(Error::Window)
    );
    assert_eq!(
        series_std(&x, Window::Observations { size: 2, ramp: 7 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        series_std(&x, Window::Observations { size: 10, ramp: 10 }),
        Err(Error::Ramp)
    );
}

#[test]
fn bollinger_bands_default_and_custom_k_across_windows() {
    let dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 5),
        date(2019, 1, 6),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // Upstream test_bollinger_bands default Window::Expanding, k = None (2.0)
    let exp_def: BollingerBands = bollinger_bands(&x, Window::Expanding, None).unwrap();
    let exp_k2 = bollinger_bands(&x, Window::Expanding, Some(2.0)).unwrap();
    assert_eq!(exp_def.lower.dates(), dates.as_slice());
    assert_eq!(exp_def.upper.dates(), dates.as_slice());
    assert!(exp_def.lower.values()[0].is_nan() && exp_def.upper.values()[0].is_nan());
    let expected_low = [
        1.0857864376269049,
        1.5119661282874148,
        0.3351457844873238,
        0.6111456180001682,
        -0.34664010613630225,
    ];
    let expected_high = [
        3.914213562373095,
        3.8213672050459184,
        4.164854215512676,
        4.188854381999832,
        6.346640106136302,
    ];
    for (i, (&lo, &hi)) in expected_low.iter().zip(&expected_high).enumerate() {
        assert!((exp_def.lower.values()[i + 1] - lo).abs() < 1e-12);
        assert!((exp_def.upper.values()[i + 1] - hi).abs() < 1e-12);
    }
    assert!(exp_def
        .lower
        .values()
        .iter()
        .zip(exp_k2.lower.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(exp_def
        .upper
        .values()
        .iter()
        .zip(exp_k2.upper.values())
        .all(|(a, b)| a.to_bits() == b.to_bits()));

    // Upstream test_bollinger_bands("2d") -> Window::Days { size: 2, ramp: 2 }
    let cal2d = bollinger_bands(&x, Window::Days { size: 2, ramp: 2 }, None).unwrap();
    assert_eq!(cal2d.lower.dates(), &dates[2..]);
    assert_eq!(cal2d.upper.dates(), &dates[2..]);
    let expected_2d_low = [
        1.0857864376269049,
        -0.8284271247461903,
        -0.8284271247461903,
        0.2573593128807148,
    ];
    let expected_2d_high = [
        3.914213562373095,
        4.82842712474619,
        4.82842712474619,
        8.742640687119286,
    ];
    for i in 0..4 {
        assert!((cal2d.lower.values()[i] - expected_2d_low[i]).abs() < 1e-12);
        assert!((cal2d.upper.values()[i] - expected_2d_high[i]).abs() < 1e-12);
    }

    // Custom k = Some(1.5) with Window::Observations { size: 3, ramp: 0 }
    let obs_k15 =
        bollinger_bands(&x, Window::Observations { size: 3, ramp: 0 }, Some(1.5)).unwrap();
    assert_eq!(obs_k15.lower.dates(), dates.as_slice());
    assert!(obs_k15.lower.values()[0].is_nan() && obs_k15.upper.values()[0].is_nan());
    assert!((obs_k15.lower.values()[3] - 0.5).abs() < 1e-12);
    assert!((obs_k15.upper.values()[3] - 3.5).abs() < 1e-12);
    assert!((obs_k15.lower.values()[5] - (-0.4415838843020419)).abs() < 1e-12);
    assert!((obs_k15.upper.values()[5] - 7.108250550968709).abs() < 1e-12);

    // Observation ramp Window::Observations { size: 3, ramp: 2 }, k = 2.0
    let obs_r2 = bollinger_bands(&x, Window::Observations { size: 3, ramp: 2 }, Some(2.0)).unwrap();
    assert_eq!(obs_r2.lower.dates(), &dates[2..]);
    assert_eq!(obs_r2.upper.dates(), &dates[2..]);
    assert!((obs_r2.lower.values()[0] - 1.5119661282874148).abs() < 1e-12);
    assert!(obs_r2.lower.values()[1] == 0.0 && obs_r2.lower.values()[1].is_sign_positive());
    assert!((obs_r2.upper.values()[1] - 4.0).abs() < 1e-12);
}

#[test]
fn bollinger_bands_leap_day_calendar_windows() {
    let leap_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 2, 29),
        date(2020, 3, 2),
        date(2020, 3, 5),
    ];
    let leap = Series::try_new(leap_dates.clone(), vec![10.0, 12.0, 11.0, 15.0, 20.0]).unwrap();

    // Window::Days { size: 3, ramp: 0 } across leap day 2020-02-29
    let d3_r0 = bollinger_bands(&leap, Window::Days { size: 3, ramp: 0 }, Some(2.0)).unwrap();
    assert_eq!(d3_r0.lower.dates(), leap_dates.as_slice());
    assert_eq!(d3_r0.upper.dates(), leap_dates.as_slice());
    assert!(d3_r0.lower.values()[0].is_nan() && d3_r0.upper.values()[0].is_nan());
    assert!((d3_r0.lower.values()[1] - 8.17157287525381).abs() < 1e-12);
    assert!((d3_r0.upper.values()[1] - 13.82842712474619).abs() < 1e-12);
    assert!((d3_r0.lower.values()[2] - 9.0).abs() < 1e-12);
    assert!((d3_r0.upper.values()[2] - 13.0).abs() < 1e-12);
    assert!((d3_r0.lower.values()[3] - 7.343145750507619).abs() < 1e-12);
    assert!((d3_r0.upper.values()[3] - 18.65685424949238).abs() < 1e-12);
    assert!(d3_r0.lower.values()[4].is_nan() && d3_r0.upper.values()[4].is_nan());

    // Window::Days { size: 3, ramp: 3 } keeps 2020-03-02 and 2020-03-05
    let d3_r3 = bollinger_bands(&leap, Window::Days { size: 3, ramp: 3 }, Some(2.0)).unwrap();
    assert_eq!(d3_r3.lower.dates(), &[date(2020, 3, 2), date(2020, 3, 5)]);
    assert_eq!(d3_r3.upper.dates(), &[date(2020, 3, 2), date(2020, 3, 5)]);
    assert!((d3_r3.lower.values()[0] - 7.343145750507619).abs() < 1e-12);
    assert!((d3_r3.upper.values()[0] - 18.65685424949238).abs() < 1e-12);
    assert!(d3_r3.lower.values()[1].is_nan() && d3_r3.upper.values()[1].is_nan());

    // Window::Days { size: 1, ramp: 0 } -> all NaN; oversize ramp -> empty
    let d1 = bollinger_bands(&leap, Window::Days { size: 1, ramp: 0 }, None).unwrap();
    assert_eq!(d1.lower.dates(), leap_dates.as_slice());
    assert!(d1.lower.values().iter().all(|v| v.is_nan()));
    assert!(d1.upper.values().iter().all(|v| v.is_nan()));

    let d30 = bollinger_bands(&leap, Window::Days { size: 30, ramp: 30 }, None).unwrap();
    assert!(d30.lower.dates().is_empty() && d30.upper.dates().is_empty());
}

#[test]
fn bollinger_bands_constants_missing_infinities_k_edges_and_determinism() {
    let dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 5),
        date(2019, 1, 6),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // Constant 4.2 -> lower == upper == 4.2 for count >= 2
    let c42 = Series::try_new(dates.clone(), vec![4.2; 6]).unwrap();
    let c42_obs = bollinger_bands(&c42, Window::Observations { size: 3, ramp: 0 }, None).unwrap();
    assert!(c42_obs.lower.values()[0].is_nan() && c42_obs.upper.values()[0].is_nan());
    for i in 1..6 {
        assert!((c42_obs.lower.values()[i] - 4.2).abs() < 1e-12);
        assert!((c42_obs.upper.values()[i] - 4.2).abs() < 1e-12);
    }

    // Constant -0.0 -> lower = -0.0, upper = +0.0 for finite k > 0.0 or +0.0 (k.is_sign_positive());
    // swapped to (+0.0, -0.0) for finite k < 0.0 or -0.0 (k.is_sign_negative()).
    let cneg0 = Series::try_new(dates.clone(), vec![-0.0; 6]).unwrap();
    for w in [
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        for k in [2.0, 0.0] {
            let out = bollinger_bands(&cneg0, w, Some(k)).unwrap();
            assert!(out.lower.values()[0].is_nan() && out.upper.values()[0].is_nan());
            for i in 1..6 {
                assert!(out.lower.values()[i] == 0.0 && out.lower.values()[i].is_sign_negative());
                assert!(out.upper.values()[i] == 0.0 && out.upper.values()[i].is_sign_positive());
            }
        }
    }
    for k in [-2.0, -0.0] {
        let out =
            bollinger_bands(&cneg0, Window::Observations { size: 3, ramp: 0 }, Some(k)).unwrap();
        assert!(out.lower.values()[0].is_nan() && out.upper.values()[0].is_nan());
        for i in 1..6 {
            assert!(out.lower.values()[i] == 0.0 && out.lower.values()[i].is_sign_positive());
            assert!(out.upper.values()[i] == 0.0 && out.upper.values()[i].is_sign_negative());
        }
    }

    // Documented boundary (P1-1): mixed signed zeros [+0.0, -0.0] under Expanding and Observations
    // emit (+0.0, +0.0) via IEEE-754 +0.0 + (-0.0) = +0.0 (numeric diff 0 vs pandas GH#42064 (-0.0, +0.0)).
    let mixed_zeros = Series::try_new(dates[..2].to_vec(), vec![0.0, -0.0]).unwrap();
    for (w, k) in [
        (Window::Expanding, None),
        (Window::Observations { size: 2, ramp: 0 }, Some(2.0)),
    ] {
        let out = bollinger_bands(&mixed_zeros, w, k).unwrap();
        assert_eq!(out.lower.dates(), &dates[..2]);
        assert!(out.lower.values()[0].is_nan() && out.upper.values()[0].is_nan());
        assert!(out.lower.values()[1] == 0.0 && out.lower.values()[1].is_sign_positive());
        assert!(out.upper.values()[1] == 0.0 && out.upper.values()[1].is_sign_positive());
    }

    // Mid-series NaN and all-NaN
    let mid_nan = Series::try_new(
        dates.clone(),
        vec![3.0, f64::NAN, 5.0, f64::NAN, f64::NAN, 7.0],
    )
    .unwrap();
    let nan_obs =
        bollinger_bands(&mid_nan, Window::Observations { size: 3, ramp: 0 }, None).unwrap();
    assert!(nan_obs.lower.values()[0].is_nan() && nan_obs.lower.values()[1].is_nan());
    assert!((nan_obs.lower.values()[2] - 1.1715728752538097).abs() < 1e-12);
    assert!((nan_obs.upper.values()[2] - 6.82842712474619).abs() < 1e-12);
    assert!(nan_obs.lower.values()[3..].iter().all(|v| v.is_nan()));
    assert!(nan_obs.upper.values()[3..].iter().all(|v| v.is_nan()));

    let all_nan = Series::try_new(dates.clone(), vec![f64::NAN; 6]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        let out = bollinger_bands(&all_nan, w, None).unwrap();
        assert_eq!(out.lower.dates(), dates.as_slice());
        assert!(out.lower.values().iter().all(|v| v.is_nan()));
        assert!(out.upper.values().iter().all(|v| v.is_nan()));
    }

    // ±Inf: skipped in Observations / Expanding vs NaN in Days
    let infs = Series::try_new(
        dates,
        vec![3.0, f64::INFINITY, 5.0, 7.0, f64::NEG_INFINITY, 9.0],
    )
    .unwrap();
    let inf_obs = bollinger_bands(&infs, Window::Observations { size: 3, ramp: 0 }, None).unwrap();
    assert!(inf_obs.lower.values()[0].is_nan() && inf_obs.lower.values()[1].is_nan());
    assert!((inf_obs.lower.values()[2] - 1.1715728752538097).abs() < 1e-12);
    assert!((inf_obs.upper.values()[2] - 6.82842712474619).abs() < 1e-12);
    assert!((inf_obs.lower.values()[3] - 3.1715728752538097).abs() < 1e-12);
    assert!((inf_obs.upper.values()[3] - 8.82842712474619).abs() < 1e-12);
    assert!((inf_obs.lower.values()[5] - 5.17157287525381).abs() < 1e-12);
    assert!((inf_obs.upper.values()[5] - 10.82842712474619).abs() < 1e-12);

    let inf_days = bollinger_bands(&infs, Window::Days { size: 4, ramp: 0 }, None).unwrap();
    assert!(inf_days.lower.values().iter().all(|v| v.is_nan()));
    assert!(inf_days.upper.values().iter().all(|v| v.is_nan()));

    // k boundary values: 0.0, -0.0, negative (-2.0), NaN, +Inf, -Inf
    let w3 = Window::Observations { size: 3, ramp: 0 };
    for kz in [0.0, -0.0] {
        let out = bollinger_bands(&x, w3, Some(kz)).unwrap();
        assert!(out.lower.values()[0].is_nan() && out.upper.values()[0].is_nan());
        for i in 1..6 {
            assert!((out.lower.values()[i] - out.upper.values()[i]).abs() < 1e-12);
        }
    }
    let k_neg = bollinger_bands(&x, w3, Some(-2.0)).unwrap();
    let k_pos = bollinger_bands(&x, w3, Some(2.0)).unwrap();
    for i in 1..6 {
        assert!((k_neg.lower.values()[i] - k_pos.upper.values()[i]).abs() < 1e-12);
        assert!((k_neg.upper.values()[i] - k_pos.lower.values()[i]).abs() < 1e-12);
    }
    let k_nan = bollinger_bands(&x, w3, Some(f64::NAN)).unwrap();
    assert!(k_nan.lower.values().iter().all(|v| v.is_nan()));
    assert!(k_nan.upper.values().iter().all(|v| v.is_nan()));

    let k_inf = bollinger_bands(&x, w3, Some(f64::INFINITY)).unwrap();
    assert!(k_inf.lower.values()[0].is_nan() && k_inf.upper.values()[0].is_nan());
    assert!(k_inf.lower.values()[1..]
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_negative()));
    assert!(k_inf.upper.values()[1..]
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_positive()));

    let k_ninf = bollinger_bands(&x, w3, Some(f64::NEG_INFINITY)).unwrap();
    assert!(k_ninf.lower.values()[0].is_nan() && k_ninf.upper.values()[0].is_nan());
    assert!(k_ninf.lower.values()[1..]
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_positive()));
    assert!(k_ninf.upper.values()[1..]
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_negative()));

    // Constant series with k = ±Inf -> Inf * 0.0 = NaN (both observation and 10-day calendar
    // windows: P1-2 documents Rust's exact +0.0 std contract vs NumPy 2.2.6 ~9.36e-16 residue).
    let c42_inf = bollinger_bands(&c42, w3, Some(f64::INFINITY)).unwrap();
    assert!(c42_inf.lower.values().iter().all(|v| v.is_nan()));
    assert!(c42_inf.upper.values().iter().all(|v| v.is_nan()));

    let ten_dates: Vec<_> = (1..=10).map(|day| date(2019, 1, day)).collect();
    let c42_10 = Series::try_new(ten_dates.clone(), vec![4.2; 10]).unwrap();
    for k in [f64::INFINITY, f64::NEG_INFINITY] {
        let out = bollinger_bands(&c42_10, Window::Days { size: 10, ramp: 0 }, Some(k)).unwrap();
        assert_eq!(out.lower.dates(), ten_dates.as_slice());
        assert_eq!(out.upper.dates(), ten_dates.as_slice());
        assert!(out.lower.values().iter().all(|v| v.is_nan()));
        assert!(out.upper.values().iter().all(|v| v.is_nan()));
    }

    // Back-to-back bitwise determinism across all edge inputs
    let checks: [(&Series, Window, Option<f64>); 18] = [
        (&c42, w3, None),
        (&cneg0, w3, Some(2.0)),
        (&cneg0, w3, Some(-0.0)),
        (&cneg0, Window::Days { size: 3, ramp: 0 }, Some(2.0)),
        (&mixed_zeros, Window::Expanding, None),
        (
            &mixed_zeros,
            Window::Observations { size: 2, ramp: 0 },
            Some(2.0),
        ),
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
        (
            &c42_10,
            Window::Days { size: 10, ramp: 0 },
            Some(f64::INFINITY),
        ),
        (
            &c42_10,
            Window::Days { size: 10, ramp: 0 },
            Some(f64::NEG_INFINITY),
        ),
    ];
    for (series, window, k) in checks {
        let a = bollinger_bands(series, window, k).unwrap();
        let b = bollinger_bands(series, window, k).unwrap();
        assert_eq!(a.lower.dates(), b.lower.dates());
        assert_eq!(a.upper.dates(), b.upper.dates());
        assert!(a
            .lower
            .values()
            .iter()
            .zip(b.lower.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(a
            .upper
            .values()
            .iter()
            .zip(b.upper.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

#[test]
fn bollinger_bands_empty_singleton_oversize_ramp_and_errors() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    for (w, k) in [
        (Window::Expanding, None),
        (Window::Observations { size: 2, ramp: 0 }, Some(2.0)),
        (Window::Observations { size: 0, ramp: 0 }, Some(2.0)),
        (Window::Observations { size: 2, ramp: 5 }, Some(2.0)),
        (Window::Days { size: 0, ramp: 0 }, Some(2.0)),
        (Window::Days { size: 2, ramp: 2 }, Some(2.0)),
        (Window::Expanding, Some(f64::NAN)),
    ] {
        let out = bollinger_bands(&empty, w, k).unwrap();
        assert!(out.lower.dates().is_empty() && out.lower.values().is_empty());
        assert!(out.upper.dates().is_empty() && out.upper.values().is_empty());
    }

    let single = Series::try_new(vec![date(2020, 1, 1)], vec![5.0]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 1, ramp: 0 },
        Window::Days { size: 2, ramp: 0 },
    ] {
        let out = bollinger_bands(&single, w, None).unwrap();
        assert_eq!(out.lower.dates(), single.dates());
        assert_eq!(out.upper.dates(), single.dates());
        assert!(out.lower.values()[0].is_nan() && out.upper.values()[0].is_nan());
    }
    assert!(
        bollinger_bands(&single, Window::Observations { size: 2, ramp: 0 }, None)
            .unwrap()
            .lower
            .values()
            .is_empty()
    );
    assert!(
        bollinger_bands(&single, Window::Observations { size: 1, ramp: 1 }, None)
            .unwrap()
            .lower
            .values()
            .is_empty()
    );
    assert_eq!(
        bollinger_bands(&single, Window::Observations { size: 1, ramp: 2 }, None),
        Err(Error::Ramp)
    );

    let dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 5),
        date(2019, 1, 6),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();

    // Size-1 observation window -> all NaN
    let obs1 = bollinger_bands(&x, Window::Observations { size: 1, ramp: 0 }, None).unwrap();
    assert_eq!(obs1.lower.dates(), dates.as_slice());
    assert!(obs1.lower.values().iter().all(|v| v.is_nan()));
    assert!(obs1.upper.values().iter().all(|v| v.is_nan()));

    // Oversize observation window & ramp == len -> empty
    assert!(
        bollinger_bands(&x, Window::Observations { size: 10, ramp: 0 }, None)
            .unwrap()
            .lower
            .values()
            .is_empty()
    );
    assert!(
        bollinger_bands(&x, Window::Observations { size: 2, ramp: 6 }, None)
            .unwrap()
            .lower
            .values()
            .is_empty()
    );

    // Error::Window and Error::Ramp (including documented Days { size: 0 } rejection and precedence)
    assert_eq!(
        bollinger_bands(&x, Window::Observations { size: 0, ramp: 0 }, None),
        Err(Error::Window)
    );
    assert_eq!(
        bollinger_bands(&x, Window::Days { size: 0, ramp: 0 }, None),
        Err(Error::Window)
    );
    assert_eq!(
        bollinger_bands(&x, Window::Observations { size: 0, ramp: 10 }, None),
        Err(Error::Window)
    );
    assert_eq!(
        bollinger_bands(&x, Window::Observations { size: 2, ramp: 7 }, None),
        Err(Error::Ramp)
    );
    assert_eq!(
        bollinger_bands(&x, Window::Observations { size: 10, ramp: 10 }, None),
        Err(Error::Ramp)
    );
}

#[test]
fn cov_upstream_values_and_window_ramp_variants() {
    let dates = vec![
        date(2019, 1, 1),
        date(2019, 1, 2),
        date(2019, 1, 3),
        date(2019, 1, 4),
        date(2019, 1, 7),
        date(2019, 1, 8),
    ];
    let x = Series::try_new(dates.clone(), vec![3.0, 2.0, 3.0, 1.0, 3.0, 6.0]).unwrap();
    let y = Series::try_new(dates.clone(), vec![3.5, 1.8, 2.9, 1.2, 3.1, 5.9]).unwrap();

    // Upstream test_cov: Window::Expanding
    let exp = cov(&x, &y, Window::Expanding).unwrap();
    assert_eq!(exp.dates(), dates.as_slice());
    assert!(exp.values()[0].is_nan());
    let expected_exp = [0.85, 0.4666666666666667, 0.95, 0.825, 2.7];
    for (i, &target) in expected_exp.iter().enumerate() {
        assert!((exp.values()[i + 1] - target).abs() < 1e-12);
    }

    // Upstream test_cov: Window::Observations { size: 2, ramp: 0 }
    let obs2 = cov(&x, &y, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(obs2.dates(), dates.as_slice());
    assert!(obs2.values()[0].is_nan());
    let expected_obs2 = [0.85, 0.55, 1.7, 1.9, 4.2];
    for (i, &target) in expected_obs2.iter().enumerate() {
        assert!((obs2.values()[i + 1] - target).abs() < 1e-12);
    }

    // Upstream test_cov: Window::Days { size: 7, ramp: 0 } ("1w")
    let cal7d = cov(&x, &y, Window::Days { size: 7, ramp: 0 }).unwrap();
    assert_eq!(cal7d.dates(), dates.as_slice());
    assert!(cal7d.values()[0].is_nan());
    let expected_7d = [0.85, 0.4666666666666667, 0.95, 0.825, 3.375];
    for (i, &target) in expected_7d.iter().enumerate() {
        assert!((cal7d.values()[i + 1] - target).abs() < 1e-12);
    }

    // Observation and calendar ramps
    let obs3_r2 = cov(&x, &y, Window::Observations { size: 3, ramp: 2 }).unwrap();
    let obs3_r0 = cov(&x, &y, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert_eq!(obs3_r2.dates(), &dates[2..]);
    assert_eq!(obs3_r2.values(), &obs3_r0.values()[2..]);

    let cal3_r3 = cov(&x, &y, Window::Days { size: 3, ramp: 3 }).unwrap();
    assert_eq!(cal3_r3.dates(), &dates[3..]);
    assert!(cal3_r3.values()[1].is_nan()); // 2019-01-07 has only itself in (2019-01-04, 2019-01-07]
    assert!((cal3_r3.values()[2] - 4.2).abs() < 1e-12);

    // Size-1 observation & calendar windows -> all NaN
    for w in [
        Window::Observations { size: 1, ramp: 0 },
        Window::Days { size: 1, ramp: 0 },
    ] {
        let out = cov(&x, &y, w).unwrap();
        assert_eq!(out.dates(), dates.as_slice());
        assert!(out.values().iter().all(|v| v.is_nan()));
    }

    // Symmetry cov(x, y) == cov(y, x) on shared dates, self-covariance == std^2, and negative cov
    for w in [
        Window::Expanding,
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 7, ramp: 0 },
    ] {
        let xy = cov(&x, &y, w).unwrap();
        let yx = cov(&y, &x, w).unwrap();
        assert_eq!(xy.dates(), yx.dates());
        assert!(xy
            .values()
            .iter()
            .zip(yx.values())
            .all(|(a, b)| a.to_bits() == b.to_bits()));

        let xx = cov(&x, &x, w).unwrap();
        let sx = series_std(&x, w).unwrap();
        assert_eq!(xx.dates(), sx.dates());
        for (&v, &sd) in xx.values().iter().zip(sx.values()) {
            if sd.is_nan() {
                assert!(v.is_nan());
            } else {
                assert!((v - sd * sd).abs() < 1e-12);
            }
        }
    }

    let d5: Vec<_> = (1..=5).map(|d| date(2020, 1, d)).collect();
    let up = Series::try_new(d5.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let down = Series::try_new(d5, vec![10.0, 8.0, 6.0, 4.0, 2.0]).unwrap();
    let neg = cov(&up, &down, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!(neg.values()[0].is_nan());
    assert!((neg.values()[1] - (-1.0)).abs() < 1e-12);
    assert!(neg.values()[2..]
        .iter()
        .all(|&v| (v - (-2.0)).abs() < 1e-12));
}

#[test]
fn cov_date_alignment_and_leap_day() {
    let d = |m, day| date(2020, m, day);

    // Partial overlap: x has 3 dates (Jan 1, 3, 5), y has 5 dates (Jan 1..5).
    // Window::Expanding uses win_size = x.dates().len() == 3 across the 5-date union!
    let px = Series::try_new(vec![d(1, 1), d(1, 3), d(1, 5)], vec![1.0, 3.0, 5.0]).unwrap();
    let py = Series::try_new(
        vec![d(1, 1), d(1, 2), d(1, 3), d(1, 4), d(1, 5)],
        vec![2.0, 10.0, 6.0, 20.0, 10.0],
    )
    .unwrap();
    let part_exp = cov(&px, &py, Window::Expanding).unwrap();
    assert_eq!(part_exp.dates(), py.dates());
    assert!(part_exp.values()[0].is_nan() && part_exp.values()[1].is_nan());
    assert!((part_exp.values()[2] - 4.0).abs() < 1e-12);
    assert!(part_exp.values()[3].is_nan()); // window [Jan 2, 3, 4] has only 1 matched pair (Jan 3)
    assert!((part_exp.values()[4] - 4.0).abs() < 1e-12);

    // Window::Observations where x.len() (3) < size (5) <= union.len() (5) returns non-empty union!
    let part_obs5 = cov(&px, &py, Window::Observations { size: 5, ramp: 0 }).unwrap();
    assert_eq!(part_obs5.dates(), py.dates());
    assert!((part_obs5.values()[2] - 4.0).abs() < 1e-12);
    assert!((part_obs5.values()[3] - 4.0).abs() < 1e-12);
    assert!((part_obs5.values()[4] - 8.0).abs() < 1e-12);

    // Subset (x ⊂ y) and superset (x ⊃ y) under Observations and Days
    let sub_days = cov(&px, &py, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert_eq!(sub_days.dates(), px.dates());
    assert!(sub_days.values()[0].is_nan());
    assert!((sub_days.values()[1] - 4.0).abs() < 1e-12);
    assert!((sub_days.values()[2] - 4.0).abs() < 1e-12);

    let sup_days = cov(&py, &px, Window::Days { size: 4, ramp: 0 }).unwrap();
    assert_eq!(sup_days.dates(), py.dates());
    assert!(sup_days.values()[0].is_nan() && sup_days.values()[1].is_nan());
    // At Jan 4 (unmatched in px), (Jan 0, Jan 4] still includes matched pairs Jan 1 and Jan 3!
    assert!((sup_days.values()[2] - 4.0).abs() < 1e-12);
    assert!((sup_days.values()[3] - 4.0).abs() < 1e-12);
    assert!((sup_days.values()[4] - 4.0).abs() < 1e-12);

    // Disjoint dates: Observations/Expanding return union dates all NaN; Days returns x.dates() all NaN
    let dx = Series::try_new(vec![d(1, 1), d(1, 2), d(1, 3)], vec![1.0, 2.0, 3.0]).unwrap();
    let dy = Series::try_new(vec![d(1, 4), d(1, 5), d(1, 6)], vec![4.0, 5.0, 6.0]).unwrap();
    let disj_obs = cov(&dx, &dy, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert_eq!(disj_obs.dates().len(), 6);
    assert!(disj_obs.values().iter().all(|v| v.is_nan()));
    let disj_days = cov(&dx, &dy, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert_eq!(disj_days.dates(), dx.dates());
    assert!(disj_days.values().iter().all(|v| v.is_nan()));

    // Interleaved dates
    let ix = Series::try_new(
        vec![d(1, 1), d(1, 2), d(1, 4), d(1, 6)],
        vec![1.0, 4.0, 2.0, 8.0],
    )
    .unwrap();
    let iy = Series::try_new(
        vec![d(1, 1), d(1, 3), d(1, 4), d(1, 5), d(1, 6)],
        vec![2.0, 7.0, 5.0, 9.0, 14.0],
    )
    .unwrap();
    let inter_obs = cov(&ix, &iy, Window::Observations { size: 4, ramp: 1 }).unwrap();
    assert_eq!(
        inter_obs.dates(),
        &[d(1, 2), d(1, 3), d(1, 4), d(1, 5), d(1, 6)]
    );
    assert!(inter_obs.values()[0].is_nan() && inter_obs.values()[1].is_nan());
    assert!((inter_obs.values()[2] - 1.5).abs() < 1e-12);
    assert!(inter_obs.values()[3].is_nan());
    assert!((inter_obs.values()[4] - 27.0).abs() < 1e-12);

    let inter_days = cov(&ix, &iy, Window::Days { size: 4, ramp: 2 }).unwrap();
    assert_eq!(inter_days.dates(), &[d(1, 4), d(1, 6)]);
    assert!((inter_days.values()[0] - 1.5).abs() < 1e-12);
    assert!((inter_days.values()[1] - 27.0).abs() < 1e-12);

    // Leap day 2020-02-29 calendar windows
    let leap_dates = vec![d(2, 27), d(2, 28), d(2, 29), d(3, 2), d(3, 5)];
    let lx = Series::try_new(leap_dates.clone(), vec![1.0, 3.0, 2.0, 6.0, 10.0]).unwrap();
    let ly = Series::try_new(leap_dates.clone(), vec![2.0, 5.0, 4.0, 11.0, 18.0]).unwrap();
    let leap_3d = cov(&lx, &ly, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert_eq!(leap_3d.dates(), leap_dates.as_slice());
    assert!(leap_3d.values()[0].is_nan());
    assert!((leap_3d.values()[1] - 3.0).abs() < 1e-12);
    assert!((leap_3d.values()[2] - 1.5).abs() < 1e-12);
    // At 2020-03-02, (2020-02-28, 2020-03-02] includes leap day 2020-02-29 and 2020-03-02
    assert!((leap_3d.values()[3] - 14.0).abs() < 1e-12);
    assert!(leap_3d.values()[4].is_nan()); // 2020-03-05 is 3 days after 2020-03-02 (excluded)

    let leap_3d_r3 = cov(&lx, &ly, Window::Days { size: 3, ramp: 3 }).unwrap();
    assert_eq!(leap_3d_r3.dates(), &[d(3, 2), d(3, 5)]);
    assert!((leap_3d_r3.values()[0] - 14.0).abs() < 1e-12);
    assert!(leap_3d_r3.values()[1].is_nan());

    // Documented boundary: Series::try_new enforces strictly increasing dates on both x and y
    assert_eq!(
        Series::try_new(vec![d(1, 2), d(1, 1)], vec![1.0, 2.0]),
        Err(Error::UnsortedOrDuplicate)
    );
    assert_eq!(
        Series::try_new(vec![d(1, 1), d(1, 1)], vec![1.0, 2.0]),
        Err(Error::UnsortedOrDuplicate)
    );
}

#[test]
fn cov_empty_singleton_oversize_and_error_boundaries() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    let d5: Vec<_> = (1..=5).map(|d| date(2020, 1, d)).collect();
    let x5 = Series::try_new(d5.clone(), vec![1.0, 2.0, 4.0, 7.0, 11.0]).unwrap();
    let y5 = Series::try_new(d5.clone(), vec![2.0, 5.0, 3.0, 9.0, 12.0]).unwrap();

    // Both empty: returns empty across all windows (including documented Days ramp > 0 vs IndexError)
    for w in [
        Window::Expanding,
        Window::Observations { size: 2, ramp: 0 },
        Window::Observations { size: 0, ramp: 0 },
        Window::Observations { size: 2, ramp: 5 },
        Window::Days { size: 0, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 1 },
    ] {
        let out = cov(&empty, &empty, w).unwrap();
        assert!(out.dates().is_empty() && out.values().is_empty());
    }

    // Empty x + non-empty y
    assert_eq!(cov(&empty, &y5, Window::Expanding), Err(Error::Window));
    assert_eq!(
        cov(&empty, &y5, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&empty, &y5, Window::Observations { size: 0, ramp: 10 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&empty, &y5, Window::Observations { size: 2, ramp: 6 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        cov(&empty, &y5, Window::Observations { size: 10, ramp: 10 }),
        Err(Error::Ramp)
    );
    let ey_obs2 = cov(&empty, &y5, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(ey_obs2.dates(), d5.as_slice());
    assert!(ey_obs2.values().iter().all(|v| v.is_nan()));
    let ey_obs2_r2 = cov(&empty, &y5, Window::Observations { size: 2, ramp: 2 }).unwrap();
    assert_eq!(ey_obs2_r2.dates(), &d5[2..]);
    assert!(ey_obs2_r2.values().iter().all(|v| v.is_nan()));
    assert!(cov(&empty, &y5, Window::Observations { size: 2, ramp: 5 })
        .unwrap()
        .values()
        .is_empty());
    assert!(cov(&empty, &y5, Window::Observations { size: 10, ramp: 0 })
        .unwrap()
        .values()
        .is_empty());
    for w in [
        Window::Days { size: 0, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 1 },
    ] {
        assert!(cov(&empty, &y5, w).unwrap().values().is_empty());
    }

    // Non-empty x + empty y
    for w in [
        Window::Expanding,
        Window::Observations { size: 2, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        let out = cov(&x5, &empty, w).unwrap();
        assert_eq!(out.dates(), d5.as_slice());
        assert!(out.values().iter().all(|v| v.is_nan()));
    }
    assert_eq!(
        cov(&x5, &empty, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&x5, &empty, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&x5, &empty, Window::Observations { size: 2, ramp: 6 }),
        Err(Error::Ramp)
    );
    assert!(cov(&x5, &empty, Window::Observations { size: 10, ramp: 0 })
        .unwrap()
        .values()
        .is_empty());

    // Singletons (same date and distinct dates)
    let s1 = Series::try_new(vec![date(2020, 1, 1)], vec![5.0]).unwrap();
    let s2 = Series::try_new(vec![date(2020, 1, 2)], vec![7.0]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 1, ramp: 0 },
        Window::Days { size: 2, ramp: 0 },
    ] {
        let out = cov(&s1, &s1, w).unwrap();
        assert_eq!(out.dates(), s1.dates());
        assert!(out.values()[0].is_nan());
    }
    let s_diff = cov(&s1, &s2, Window::Observations { size: 2, ramp: 0 }).unwrap();
    assert_eq!(s_diff.dates(), &[date(2020, 1, 1), date(2020, 1, 2)]);
    assert!(s_diff.values().iter().all(|v| v.is_nan()));
    assert!(cov(&s1, &s1, Window::Observations { size: 2, ramp: 0 })
        .unwrap()
        .values()
        .is_empty());

    // Oversize windows, ramp == len, Days oversize ramp, and error precedence
    assert!(cov(&x5, &y5, Window::Observations { size: 10, ramp: 0 })
        .unwrap()
        .values()
        .is_empty());
    assert!(cov(&x5, &y5, Window::Observations { size: 2, ramp: 5 })
        .unwrap()
        .values()
        .is_empty());
    assert!(cov(&x5, &y5, Window::Days { size: 30, ramp: 30 })
        .unwrap()
        .values()
        .is_empty());
    assert_eq!(
        cov(&x5, &y5, Window::Observations { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&x5, &y5, Window::Days { size: 0, ramp: 0 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&x5, &y5, Window::Observations { size: 0, ramp: 10 }),
        Err(Error::Window)
    );
    assert_eq!(
        cov(&x5, &y5, Window::Observations { size: 2, ramp: 6 }),
        Err(Error::Ramp)
    );
    assert_eq!(
        cov(&x5, &y5, Window::Observations { size: 10, ramp: 10 }),
        Err(Error::Ramp)
    );
    // Partial overlap where ramp > x.len() (3) even though ramp <= union.len() (5) -> Error::Ramp
    let x3 = Series::try_new(vec![d5[0], d5[2], d5[4]], vec![1.0, 3.0, 5.0]).unwrap();
    assert_eq!(
        cov(&x3, &y5, Window::Observations { size: 4, ramp: 4 }),
        Err(Error::Ramp)
    );
}

#[test]
fn cov_nan_infinities_constants_signed_zeros_large_offset_overflow_and_determinism() {
    let d5: Vec<_> = (1..=5).map(|d| date(2020, 1, d)).collect();
    let lin5 = Series::try_new(d5.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

    // NaN handling across Observations and Days
    let x_nan = Series::try_new(d5.clone(), vec![1.0, f64::NAN, 3.0, 4.0, 5.0]).unwrap();
    let y_nan = Series::try_new(d5.clone(), vec![2.0, 4.0, f64::NAN, 8.0, 10.0]).unwrap();
    for w in [
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        let out = cov(&x_nan, &y_nan, w).unwrap();
        assert!(out.values()[..4].iter().all(|v| v.is_nan()));
        assert!((out.values()[4] - 1.0).abs() < 1e-12);
    }
    let all_nan = Series::try_new(d5.clone(), vec![f64::NAN; 5]).unwrap();
    for w in [
        Window::Expanding,
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        let out = cov(&all_nan, &lin5, w).unwrap();
        assert!(out.values().iter().all(|v| v.is_nan()));
    }

    // ±Inf: skipped in Expanding / Observations; in Days, Inf paired with finite/Inf -> NaN,
    // while Inf paired with NaN or an unmatched date is dropped first!
    let x_inf = Series::try_new(d5.clone(), vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0]).unwrap();
    let y_fin = Series::try_new(d5.clone(), vec![2.0, 4.0, 6.0, 8.0, 10.0]).unwrap();
    let y_nan_at_inf = Series::try_new(d5.clone(), vec![2.0, f64::NAN, 6.0, 8.0, 10.0]).unwrap();
    let y_unmatched =
        Series::try_new(vec![d5[0], d5[2], d5[3], d5[4]], vec![2.0, 6.0, 8.0, 10.0]).unwrap();

    let inf_obs = cov(&x_inf, &y_fin, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!(inf_obs.values()[0].is_nan() && inf_obs.values()[1].is_nan());
    assert!((inf_obs.values()[2] - 4.0).abs() < 1e-12);
    assert!((inf_obs.values()[3] - 1.0).abs() < 1e-12);
    assert!((inf_obs.values()[4] - 2.0).abs() < 1e-12);

    let inf_days_fin = cov(&x_inf, &y_fin, Window::Days { size: 3, ramp: 0 }).unwrap();
    assert!(inf_days_fin.values()[..4].iter().all(|v| v.is_nan()));
    assert!((inf_days_fin.values()[4] - 2.0).abs() < 1e-12);

    for y_drop in [&y_nan_at_inf, &y_unmatched] {
        let out = cov(&x_inf, y_drop, Window::Days { size: 3, ramp: 0 }).unwrap();
        assert!(out.values()[0].is_nan() && out.values()[1].is_nan());
        assert!((out.values()[2] - 4.0).abs() < 1e-12);
        assert!((out.values()[3] - 1.0).abs() < 1e-12);
        assert!((out.values()[4] - 2.0).abs() < 1e-12);
    }

    // Identical [+Inf, +Inf] paired with finite in Days must return NaN (not +0.0)
    let all_inf = Series::try_new(d5[..2].to_vec(), vec![f64::INFINITY, f64::INFINITY]).unwrap();
    let fin2 = Series::try_new(d5[..2].to_vec(), vec![1.0, 1.0]).unwrap();
    assert!(cov(&all_inf, &fin2, Window::Days { size: 2, ramp: 0 })
        .unwrap()
        .values()
        .iter()
        .all(|v| v.is_nan()));

    // Constants (4.2, 0.1, -0.0, mixed signed zeros) and orthogonal non-constant pairs -> exact +0.0
    let c42 = Series::try_new(d5.clone(), vec![4.2; 5]).unwrap();
    let c01 = Series::try_new(d5.clone(), vec![0.1; 5]).unwrap();
    let cneg0 = Series::try_new(d5.clone(), vec![-0.0; 5]).unwrap();
    let cmix0 = Series::try_new(d5.clone(), vec![-0.0, 0.0, -0.0, 0.0, -0.0]).unwrap();
    for (a, b, w) in [
        (&c42, &lin5, Window::Observations { size: 3, ramp: 0 }),
        (&lin5, &c42, Window::Observations { size: 3, ramp: 0 }),
        (&c01, &c01, Window::Days { size: 3, ramp: 0 }),
        (&cneg0, &cneg0, Window::Observations { size: 3, ramp: 0 }),
        (&cneg0, &lin5, Window::Days { size: 3, ramp: 0 }),
        (&cmix0, &lin5, Window::Observations { size: 3, ramp: 0 }),
    ] {
        let out = cov(a, b, w).unwrap();
        assert!(out.values()[0].is_nan());
        for &v in &out.values()[1..] {
            assert!(v == 0.0 && v.is_sign_positive());
        }
    }
    let ox = Series::try_new(d5[..4].to_vec(), vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    let oy = Series::try_new(d5[..4].to_vec(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let ortho = cov(&ox, &oy, Window::Observations { size: 3, ramp: 0 }).unwrap();
    assert!(ortho.values()[2] == 0.0 && ortho.values()[2].is_sign_positive());
    assert!(ortho.values()[3] == 0.0 && ortho.values()[3].is_sign_positive());

    // Large-offset stability: two-pass centered covariance avoids pandas rolling().cov() cancellation
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
    let expected_lo3 = [1.5, 1.0 / 3.0, 17.0 / 3.0, 15.5];
    for w in [
        Window::Observations { size: 3, ramp: 0 },
        Window::Days { size: 3, ramp: 0 },
    ] {
        let out = cov(&lo_x, &lo_y, w).unwrap();
        assert!(out.values()[0].is_nan());
        for (i, &target) in expected_lo3.iter().enumerate() {
            assert!((out.values()[i + 1] - target).abs() < 1e-12);
        }
    }
    let lo_exp = cov(&lo_x, &lo_y, Window::Expanding).unwrap();
    let expected_lo_exp = [1.5, 1.0 / 3.0, 41.0 / 6.0, 16.0];
    for (i, &target) in expected_lo_exp.iter().enumerate() {
        assert!((lo_exp.values()[i + 1] - target).abs() < 1e-12);
    }

    // Overflow ([1e308, -1e308]) and large same-sign overflow-safe mean fallback
    let d2 = d5[..2].to_vec();
    let ov_x = Series::try_new(d2.clone(), vec![1e308, -1e308]).unwrap();
    let ov_y = Series::try_new(d2.clone(), vec![1e308, -1e308]).unwrap();
    let ov_yn = Series::try_new(d2.clone(), vec![-1e308, 1e308]).unwrap();
    for w in [
        Window::Observations { size: 2, ramp: 0 },
        Window::Days { size: 2, ramp: 0 },
    ] {
        let pos = cov(&ov_x, &ov_y, w).unwrap();
        assert!(pos.values()[0].is_nan());
        assert!(pos.values()[1].is_infinite() && pos.values()[1].is_sign_positive());
        let neg = cov(&ov_x, &ov_yn, w).unwrap();
        assert!(neg.values()[0].is_nan());
        assert!(neg.values()[1].is_infinite() && neg.values()[1].is_sign_negative());
    }
    let big_same_x = Series::try_new(d2.clone(), vec![1e308, 1.2e308]).unwrap();
    let big_same_y = Series::try_new(d2.clone(), vec![1.2e308, 1e308]).unwrap();
    let small_y = Series::try_new(d2, vec![1.0, 2.0]).unwrap();
    let big_anti = cov(
        &big_same_x,
        &big_same_y,
        Window::Observations { size: 2, ramp: 0 },
    )
    .unwrap();
    assert!(big_anti.values()[1].is_infinite() && big_anti.values()[1].is_sign_negative());
    let big_fin = cov(
        &big_same_x,
        &small_y,
        Window::Observations { size: 2, ramp: 0 },
    )
    .unwrap();
    assert!((big_fin.values()[1] - 1e307).abs() / 1e307 < 1e-12);

    // Back-to-back bitwise determinism across all edge inputs
    let checks: [(&Series, &Series, Window); 10] = [
        (&x_nan, &y_nan, Window::Observations { size: 3, ramp: 0 }),
        (&x_inf, &y_fin, Window::Observations { size: 3, ramp: 0 }),
        (&x_inf, &y_fin, Window::Days { size: 3, ramp: 0 }),
        (&x_inf, &y_nan_at_inf, Window::Days { size: 3, ramp: 0 }),
        (&c42, &lin5, Window::Observations { size: 3, ramp: 0 }),
        (&c01, &c01, Window::Days { size: 3, ramp: 0 }),
        (&cneg0, &cneg0, Window::Observations { size: 3, ramp: 0 }),
        (&lo_x, &lo_y, Window::Observations { size: 3, ramp: 0 }),
        (&ov_x, &ov_y, Window::Observations { size: 2, ramp: 0 }),
        (
            &big_same_x,
            &small_y,
            Window::Observations { size: 2, ramp: 0 },
        ),
    ];
    for (a, b, w) in checks {
        let r1 = cov(a, b, w).unwrap();
        let r2 = cov(a, b, w).unwrap();
        assert_eq!(r1.dates(), r2.dates());
        assert!(r1
            .values()
            .iter()
            .zip(r2.values())
            .all(|(u, v)| u.to_bits() == v.to_bits()));
    }
}

#[test]
fn rolling_linear_regression_upstream_and_multi_factor() {
    let d_jan = |day| date(2019, 1, day);

    // 1. Upstream test_statistics.py::test_rolling_linear_regression
    let x1 = Series::try_new(
        (1..=7).map(d_jan).collect(),
        vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, f64::NAN],
    )
    .unwrap();
    let x2 = Series::try_new(
        (1..=8).map(d_jan).collect(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    .unwrap();
    let y = Series::try_new(
        (1..=7).map(d_jan).collect(),
        vec![10.0, 14.0, 20.0, 28.0, 28.0, 40.0, 60.0],
    )
    .unwrap();

    let reg = RollingLinearRegression::fit(&[x1.clone(), x2.clone()], &y, 4, true).unwrap();
    let expected_dates: Vec<_> = (1..=6).map(d_jan).collect();
    assert_eq!(reg.column_indices(), &[0, 1, 2]);
    assert_eq!(reg.r_squared().dates(), expected_dates.as_slice());
    assert_eq!(reg.fitted_values().dates(), expected_dates.as_slice());
    assert_eq!(
        reg.standard_deviation_of_errors().dates(),
        expected_dates.as_slice()
    );

    let assert_ols_series = |actual: &Series, expected: &[f64]| {
        assert_eq!(actual.dates(), expected_dates.as_slice());
        for (&a, &e) in actual.values().iter().zip(expected) {
            if e.is_nan() {
                assert!(a.is_nan(), "expected NaN, got {a}");
            } else {
                assert!(
                    (a - e).abs() <= 1e-10 + 1e-8 * e.abs(),
                    "actual={a}, expected={e}"
                );
            }
        }
    };
    assert_ols_series(
        reg.coefficient(0).unwrap(),
        &[f64::NAN, f64::NAN, f64::NAN, 10.0, 2.5, 19.0],
    );
    assert_ols_series(
        reg.coefficient(1).unwrap(),
        &[f64::NAN, f64::NAN, f64::NAN, 1.0, -1.5, 1.0],
    );
    assert_ols_series(
        reg.coefficient(2).unwrap(),
        &[f64::NAN, f64::NAN, f64::NAN, 3.0, 12.5, -1.0],
    );
    assert_ols_series(
        reg.r_squared(),
        &[
            f64::NAN,
            f64::NAN,
            f64::NAN,
            1.0,
            0.9640287769784173,
            0.9019607843137254,
        ],
    );
    assert_ols_series(
        reg.fitted_values(),
        &[f64::NAN, f64::NAN, f64::NAN, 28.0, 28.5, 39.0],
    );
    assert_ols_series(
        reg.standard_deviation_of_errors(),
        &[
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0.0,
            2.236067977499789,
            4.4721359549995805,
        ],
    );

    // 2. Same upstream data with fit_intercept = false (column_indices = [1, 2], uncentered R^2)
    let reg_no_int = RollingLinearRegression::fit(&[x1.clone(), x2.clone()], &y, 4, false).unwrap();
    assert_eq!(reg_no_int.column_indices(), &[1, 2]);
    assert!(reg_no_int.coefficient(0).is_none());
    assert!(reg_no_int.coefficient(3).is_none());
    assert!(reg_no_int.r_squared().values()[..3]
        .iter()
        .all(|v| v.is_nan()));
    // Window [2..6]: x1=[4,9,16,25], x2=[2,3,4,5], y=[20,28,28,40]; uncentered R^2 finite in (0,1)
    assert!(reg_no_int.r_squared().values()[5] > 0.9 && reg_no_int.r_squared().values()[5] < 1.0);
    assert!(reg_no_int.standard_deviation_of_errors().values()[5] > 0.0);

    // 3. 1-predictor models (with and without intercept, including minimum window k+1)
    let d6: Vec<_> = (1..=6).map(|day| date(2020, 1, day)).collect();
    let p1 = Series::try_new(d6.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y1 = Series::try_new(d6.clone(), vec![3.0, 5.0, 7.0, 11.0, 11.0, 13.0]).unwrap();
    let one_int = RollingLinearRegression::fit(std::slice::from_ref(&p1), &y1, 3, true).unwrap();
    assert_eq!(one_int.column_indices(), &[0, 1]);
    assert!(one_int.coefficient(0).unwrap().values()[..2]
        .iter()
        .all(|v| v.is_nan()));
    // First window [1,2,3] -> [3,5,7]: exact intercept=1, slope=2, R^2=1, std_err=0
    assert!((one_int.coefficient(0).unwrap().values()[2] - 1.0).abs() < 1e-10);
    assert!((one_int.coefficient(1).unwrap().values()[2] - 2.0).abs() < 1e-10);
    assert!((one_int.r_squared().values()[2] - 1.0).abs() < 1e-10);
    assert!(one_int.standard_deviation_of_errors().values()[2].abs() < 1e-10);

    let one_noint = RollingLinearRegression::fit(std::slice::from_ref(&p1), &y1, 2, false).unwrap();
    assert_eq!(one_noint.column_indices(), &[1]);
    assert!(one_noint.coefficient(1).unwrap().values()[0].is_nan());
    // Window [1, 2] -> [3, 5]: beta = (1*3 + 2*5)/(1 + 4) = 13/5 = 2.6
    assert!((one_noint.coefficient(1).unwrap().values()[1] - 2.6).abs() < 1e-10);
    assert!((one_noint.fitted_values().values()[1] - 5.2).abs() < 1e-10);

    // 4. 3-predictor models (with and without intercept, and window == aligned.len())
    let d7: Vec<_> = (1..=7).map(|day| date(2020, 3, day)).collect();
    let f1 = Series::try_new(d7.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let f2 = Series::try_new(d7.clone(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 1.0]).unwrap();
    let f3 = Series::try_new(d7.clone(), vec![2.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0]).unwrap();
    let y3 = Series::try_new(d7.clone(), vec![9.0, 11.0, 18.0, 22.0, 29.0, 27.0, 34.0]).unwrap();
    let three_int =
        RollingLinearRegression::fit(&[f1.clone(), f2.clone(), f3.clone()], &y3, 5, true).unwrap();
    assert_eq!(three_int.column_indices(), &[0, 1, 2, 3]);
    assert!(three_int.r_squared().values()[..4]
        .iter()
        .all(|v| v.is_nan()));
    assert!(three_int.r_squared().values()[4..]
        .iter()
        .all(|v| v.is_finite()));

    let three_noint =
        RollingLinearRegression::fit(&[f1.clone(), f2.clone(), f3.clone()], &y3, 4, false).unwrap();
    assert_eq!(three_noint.column_indices(), &[1, 2, 3]);
    assert!(three_noint.r_squared().values()[..3]
        .iter()
        .all(|v| v.is_nan()));
    assert!(three_noint.r_squared().values()[3..]
        .iter()
        .all(|v| v.is_finite()));

    // Exact window == aligned.len(): only the final row is populated
    let exact_len = RollingLinearRegression::fit(&[f1, f2, f3], &y3, 7, true).unwrap();
    assert!(exact_len.r_squared().values()[..6]
        .iter()
        .all(|v| v.is_nan()));
    assert!(exact_len.r_squared().values()[6].is_finite());
}

#[test]
fn rolling_linear_regression_constant_skip_centered_r2_and_rank_deficiency() {
    let d6: Vec<_> = (1..=6).map(|day| date(2020, 4, day)).collect();
    let x1 = Series::try_new(d6.clone(), vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0]).unwrap();
    let c5 = Series::try_new(d6.clone(), vec![5.0; 6]).unwrap();
    let c0 = Series::try_new(d6.clone(), vec![0.0; 6]).unwrap();
    let y = Series::try_new(d6.clone(), vec![12.0, 14.0, 19.0, 24.0, 33.0, 42.0]).unwrap();

    // 1. sm.add_constant skips prepending 1.0 when a predictor is globally non-zero constant
    let skipped = RollingLinearRegression::fit(&[x1.clone(), c5.clone()], &y, 3, true).unwrap();
    assert_eq!(skipped.column_indices(), &[0, 1]);
    let with_auto = RollingLinearRegression::fit(std::slice::from_ref(&x1), &y, 3, true).unwrap();
    for idx in 2..6 {
        // Slope matches and constant-5 coefficient equals intercept / 5
        assert!(
            (skipped.coefficient(0).unwrap().values()[idx]
                - with_auto.coefficient(1).unwrap().values()[idx])
                .abs()
                < 1e-10
        );
        assert!(
            (skipped.coefficient(1).unwrap().values()[idx] * 5.0
                - with_auto.coefficient(0).unwrap().values()[idx])
                .abs()
                < 1e-10
        );
        assert!(
            (skipped.r_squared().values()[idx] - with_auto.r_squared().values()[idx]).abs() < 1e-10
        );
    }

    // 2. All-zero predictor is NOT skipped by sm.add_constant -> prepends 1.0 -> rank-deficient
    let zero_pred = RollingLinearRegression::fit(&[x1.clone(), c0], &y, 4, true).unwrap();
    assert_eq!(zero_pred.column_indices(), &[0, 1, 2]);
    assert!(zero_pred.r_squared().values().iter().all(|v| v.is_nan()));
    assert!(zero_pred
        .fitted_values()
        .values()
        .iter()
        .all(|v| v.is_nan()));

    // 3. Constant predictor with NaN or differing outside y is NOT skipped -> prepends 1.0 -> collinear
    let c5_nan = Series::try_new(d6.clone(), vec![5.0, 5.0, f64::NAN, 5.0, 5.0, 5.0]).unwrap();
    let nan_no_skip = RollingLinearRegression::fit(&[x1.clone(), c5_nan], &y, 4, true).unwrap();
    assert_eq!(nan_no_skip.column_indices(), &[0, 1, 2]);
    assert_eq!(nan_no_skip.r_squared().dates().len(), 5);
    assert!(nan_no_skip.r_squared().values().iter().all(|v| v.is_nan()));

    let d7: Vec<_> = (1..=7).map(|day| date(2020, 4, day)).collect();
    let c5_extra = Series::try_new(d7, vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0]).unwrap();
    let extra_no_skip =
        RollingLinearRegression::fit(&[x1.clone(), c5_extra.clone()], &y, 4, true).unwrap();
    assert_eq!(extra_no_skip.column_indices(), &[0, 1, 2]);
    assert_eq!(extra_no_skip.r_squared().dates().len(), 6);
    for &col in extra_no_skip.column_indices() {
        assert!(extra_no_skip
            .coefficient(col)
            .unwrap()
            .values()
            .iter()
            .all(|v| v.is_nan()));
    }
    assert!(extra_no_skip
        .r_squared()
        .values()
        .iter()
        .all(|v| v.is_nan()));
    assert!(extra_no_skip
        .fitted_values()
        .values()
        .iter()
        .all(|v| v.is_nan()));
    assert!(extra_no_skip
        .standard_deviation_of_errors()
        .values()
        .iter()
        .all(|v| v.is_nan()));

    // 4. has_constant: fit_intercept = false with globally constant predictor on aligned dates uses centered R^2
    let noint_const = RollingLinearRegression::fit(&[x1.clone(), c5], &y, 4, false).unwrap();
    let noint_const_aligned =
        RollingLinearRegression::fit(&[x1.clone(), c5_extra], &y, 4, false).unwrap();
    let auto_w4 = RollingLinearRegression::fit(std::slice::from_ref(&x1), &y, 4, true).unwrap();
    let noint_plain =
        RollingLinearRegression::fit(std::slice::from_ref(&x1), &y, 4, false).unwrap();
    assert_eq!(noint_const.column_indices(), &[1, 2]);
    assert_eq!(noint_const_aligned.column_indices(), &[1, 2]);
    for idx in 3..6 {
        // Centered R^2 matches fit_intercept = true
        assert!(
            (noint_const.r_squared().values()[idx] - auto_w4.r_squared().values()[idx]).abs()
                < 1e-10
        );
        assert!(
            (noint_const_aligned.r_squared().values()[idx] - auto_w4.r_squared().values()[idx])
                .abs()
                < 1e-10
        );
        // Uncentered R^2 without any constant differs from centered R^2
        assert!(
            (noint_plain.r_squared().values()[idx] - auto_w4.r_squared().values()[idx]).abs()
                > 1e-3
        );
    }

    // 5. Window-local constant predictor: first window NaN, later windows full-rank
    let local_const = Series::try_new(d6.clone(), vec![2.0, 2.0, 2.0, 2.0, 5.0, 9.0]).unwrap();
    let lc = RollingLinearRegression::fit(&[local_const], &y, 4, true).unwrap();
    assert!(lc.r_squared().values()[..4].iter().all(|v| v.is_nan()));
    assert!(lc.coefficient(0).unwrap().values()[3].is_nan());
    assert!(lc.r_squared().values()[4].is_finite());
    assert!(lc.r_squared().values()[5].is_finite());

    // 6. Collinear predictors: full collinearity -> all NaN; partial collinearity -> early NaN then finite
    let collin_all = Series::try_new(
        d6.clone(),
        x1.values().iter().map(|&v| 2.0 * v + 3.0).collect(),
    )
    .unwrap();
    let col_all_reg = RollingLinearRegression::fit(&[x1.clone(), collin_all], &y, 4, true).unwrap();
    assert_eq!(col_all_reg.column_indices(), &[0, 1, 2]);
    for &col in col_all_reg.column_indices() {
        assert!(col_all_reg
            .coefficient(col)
            .unwrap()
            .values()
            .iter()
            .all(|v| v.is_nan()));
    }
    assert!(col_all_reg.r_squared().values().iter().all(|v| v.is_nan()));
    assert!(col_all_reg
        .fitted_values()
        .values()
        .iter()
        .all(|v| v.is_nan()));
    assert!(col_all_reg
        .standard_deviation_of_errors()
        .values()
        .iter()
        .all(|v| v.is_nan()));

    let collin_part = Series::try_new(d6, vec![2.0, 4.0, 8.0, 14.0, 30.0, 50.0]).unwrap();
    let col_part_reg = RollingLinearRegression::fit(&[x1, collin_part], &y, 4, true).unwrap();
    assert!(col_part_reg.r_squared().values()[3].is_nan());
    assert!(col_part_reg.r_squared().values()[4].is_finite());
    assert!(col_part_reg.r_squared().values()[5].is_finite());
}

#[test]
fn rolling_linear_regression_filtering_alignment_leap_day_and_errors() {
    let d = |m, day| date(2020, m, day);

    // 1. NaN / ±Inf row filtering across predictors and y
    let dates8: Vec<_> = (1..=8).map(|day| d(5, day)).collect();
    let x1 = Series::try_new(
        dates8.clone(),
        vec![1.0, f64::NAN, 2.0, 3.0, f64::INFINITY, 5.0, 7.0, 9.0],
    )
    .unwrap();
    let x2 = Series::try_new(
        dates8.clone(),
        vec![2.0, 1.0, 5.0, f64::NEG_INFINITY, 4.0, 3.0, 8.0, 6.0],
    )
    .unwrap();
    let y = Series::try_new(
        dates8.clone(),
        vec![5.0, 6.0, 10.0, 11.0, 12.0, f64::INFINITY, 25.0, 27.0],
    )
    .unwrap();
    // Surviving dates: May 1, May 3, May 7, May 8 (4 rows)
    let filt = RollingLinearRegression::fit(&[x1, x2], &y, 4, true).unwrap();
    assert_eq!(
        filt.r_squared().dates(),
        &[d(5, 1), d(5, 3), d(5, 7), d(5, 8)]
    );
    assert!(filt.r_squared().values()[..3].iter().all(|v| v.is_nan()));
    assert!(filt.r_squared().values()[3].is_finite());

    // 2. Misaligned dates (outer-join predictors, inner-join with y)
    let mx1 = Series::try_new(
        vec![d(6, 1), d(6, 2), d(6, 3), d(6, 4), d(6, 5), d(6, 6)],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let mx2 = Series::try_new(
        vec![d(6, 1), d(6, 3), d(6, 4), d(6, 5), d(6, 6), d(6, 7)],
        vec![2.0, 1.0, 4.0, 2.0, 5.0, 9.0],
    )
    .unwrap();
    let my = Series::try_new(
        vec![d(6, 1), d(6, 2), d(6, 3), d(6, 4), d(6, 5), d(6, 7)],
        vec![4.0, 99.0, 8.0, 13.0, 14.0, 88.0],
    )
    .unwrap();
    // Complete finite aligned dates: Jun 1, Jun 3, Jun 4, Jun 5
    let aligned = RollingLinearRegression::fit(&[mx1.clone(), mx2], &my, 4, true).unwrap();
    assert_eq!(
        aligned.r_squared().dates(),
        &[d(6, 1), d(6, 3), d(6, 4), d(6, 5)]
    );
    assert!(aligned.r_squared().values()[3].is_finite());

    // 3. Irregular dates crossing leap day 2020-02-29
    let leap_dates = vec![d(2, 26), d(2, 28), d(2, 29), d(3, 2), d(3, 5), d(3, 9)];
    let lx = Series::try_new(leap_dates.clone(), vec![1.0, 2.0, 4.0, 5.0, 8.0, 10.0]).unwrap();
    let ly = Series::try_new(leap_dates.clone(), vec![2.0, 5.0, 8.0, 11.0, 15.0, 21.0]).unwrap();
    let leap_reg = RollingLinearRegression::fit(std::slice::from_ref(&lx), &ly, 3, true).unwrap();
    assert_eq!(leap_reg.r_squared().dates(), leap_dates.as_slice());
    assert!(leap_reg.r_squared().values()[2..]
        .iter()
        .all(|v| v.is_finite()));

    // 4. Error boundaries
    let empty = Series::try_new(vec![], vec![]).unwrap();
    // Empty predictors -> Error::Length (even when window == 0)
    assert_eq!(
        RollingLinearRegression::fit(&[], &ly, 4, true),
        Err(Error::Length)
    );
    assert_eq!(
        RollingLinearRegression::fit(&[], &ly, 0, false),
        Err(Error::Length)
    );
    // window <= k_cols -> Error::Window
    assert_eq!(
        RollingLinearRegression::fit(std::slice::from_ref(&lx), &ly, 0, true),
        Err(Error::Window)
    );
    assert_eq!(
        RollingLinearRegression::fit(std::slice::from_ref(&lx), &ly, 2, true),
        Err(Error::Window)
    );
    assert_eq!(
        RollingLinearRegression::fit(std::slice::from_ref(&lx), &ly, 1, false),
        Err(Error::Window)
    );
    let c5 = Series::try_new(leap_dates, vec![5.0; 6]).unwrap();
    // With const skip, k_cols = 2, so window = 2 fails while window = 3 succeeds
    assert_eq!(
        RollingLinearRegression::fit(&[lx.clone(), c5.clone()], &ly, 2, true),
        Err(Error::Window)
    );
    assert!(RollingLinearRegression::fit(&[lx.clone(), c5], &ly, 3, true).is_ok());

    // Short aligned series, empty series, and disjoint dates -> Error::Window
    assert_eq!(
        RollingLinearRegression::fit(std::slice::from_ref(&lx), &ly, 7, true),
        Err(Error::Window)
    );
    assert_eq!(
        RollingLinearRegression::fit(std::slice::from_ref(&empty), &empty, 3, true),
        Err(Error::Window)
    );
    let disj_y = Series::try_new(
        vec![d(8, 1), d(8, 2), d(8, 3), d(8, 4)],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    assert_eq!(
        RollingLinearRegression::fit(&[mx1], &disj_y, 3, true),
        Err(Error::Window)
    );
    // Short after NaN filtering
    let short_y = Series::try_new(
        vec![d(2, 26), d(2, 28), d(2, 29), d(3, 2), d(3, 5), d(3, 9)],
        vec![1.0, f64::NAN, f64::INFINITY, 4.0, f64::NEG_INFINITY, 6.0],
    )
    .unwrap();
    assert_eq!(
        RollingLinearRegression::fit(&[lx], &short_y, 4, true),
        Err(Error::Window)
    );
}

#[test]
fn rolling_linear_regression_const_zero_y_perfect_fit_ill_conditioned_and_determinism() {
    let d6: Vec<_> = (1..=6).map(|day| date(2020, 7, day)).collect();
    let x = Series::try_new(d6.clone(), vec![0.1, 0.3, 0.7, 1.3, 2.1, 3.1]).unwrap();

    // 1. Constant y with intercept: exact beta=[5.3, 0.0], fitted=5.3, r_squared=NaN, std_err=+0.0
    let y_const = Series::try_new(d6.clone(), vec![5.3; 6]).unwrap();
    let const_reg =
        RollingLinearRegression::fit(std::slice::from_ref(&x), &y_const, 4, true).unwrap();
    for idx in 3..6 {
        assert_eq!(const_reg.coefficient(0).unwrap().values()[idx], 5.3);
        assert_eq!(const_reg.coefficient(1).unwrap().values()[idx], 0.0);
        assert_eq!(const_reg.fitted_values().values()[idx], 5.3);
        assert!(const_reg.r_squared().values()[idx].is_nan());
        let se = const_reg.standard_deviation_of_errors().values()[idx];
        assert!(se == 0.0 && se.is_sign_positive());
    }
    // Constant y on a rank-deficient window must still yield all NaN
    let x_local_const = Series::try_new(d6.clone(), vec![2.0, 2.0, 2.0, 2.0, 5.0, 9.0]).unwrap();
    let const_rank_def = RollingLinearRegression::fit(&[x_local_const], &y_const, 4, true).unwrap();
    assert!(const_rank_def.coefficient(0).unwrap().values()[3].is_nan());
    assert!(const_rank_def.fitted_values().values()[3].is_nan());
    assert!(const_rank_def.standard_deviation_of_errors().values()[3].is_nan());
    assert_eq!(const_rank_def.coefficient(0).unwrap().values()[4], 5.3);
    assert_eq!(
        const_rank_def.standard_deviation_of_errors().values()[4],
        0.0
    );

    // 2. Zero y (with and without intercept) -> beta=0, fitted=0, std_err=+0.0, r_squared=NaN
    let y_zero = Series::try_new(d6.clone(), vec![0.0; 6]).unwrap();
    for fit_intercept in [true, false] {
        let zreg =
            RollingLinearRegression::fit(std::slice::from_ref(&x), &y_zero, 4, fit_intercept)
                .unwrap();
        for idx in 3..6 {
            for &col in zreg.column_indices() {
                assert_eq!(zreg.coefficient(col).unwrap().values()[idx], 0.0);
            }
            assert_eq!(zreg.fitted_values().values()[idx], 0.0);
            assert!(zreg.r_squared().values()[idx].is_nan());
            let se = zreg.standard_deviation_of_errors().values()[idx];
            assert!(se == 0.0 && se.is_sign_positive());
        }
    }

    // 3. Perfect fit (with and without intercept)
    let x2 = Series::try_new(d6.clone(), vec![2.0, 1.0, 4.0, 2.0, 5.0, 3.0]).unwrap();
    let y_perf = Series::try_new(
        d6.clone(),
        x.values()
            .iter()
            .zip(x2.values())
            .map(|(&a, &b)| 3.0 + 2.0 * a - 0.5 * b)
            .collect(),
    )
    .unwrap();
    let perf = RollingLinearRegression::fit(&[x.clone(), x2.clone()], &y_perf, 4, true).unwrap();
    for idx in 3..6 {
        assert!((perf.coefficient(0).unwrap().values()[idx] - 3.0).abs() < 1e-10);
        assert!((perf.coefficient(1).unwrap().values()[idx] - 2.0).abs() < 1e-10);
        assert!((perf.coefficient(2).unwrap().values()[idx] - (-0.5)).abs() < 1e-10);
        assert!((perf.r_squared().values()[idx] - 1.0).abs() < 1e-10);
        assert!((perf.fitted_values().values()[idx] - y_perf.values()[idx]).abs() < 1e-10);
        assert!(perf.standard_deviation_of_errors().values()[idx] < 1e-10);
    }

    // 4. Ill-conditioned predictor (1e7 + [1..6]) with intercept: Householder QR preserves accuracy
    let x_ill = Series::try_new(d6.clone(), (1..=6).map(|k| 1e7 + k as f64).collect()).unwrap();
    let y_ill = Series::try_new(d6, (1..=6).map(|k| 1.0 + 2.0 * k as f64).collect()).unwrap();
    let ill = RollingLinearRegression::fit(std::slice::from_ref(&x_ill), &y_ill, 4, true).unwrap();
    for idx in 3..6 {
        let intercept = ill.coefficient(0).unwrap().values()[idx];
        let slope = ill.coefficient(1).unwrap().values()[idx];
        // True model: y = -19999999.0 + 2.0 * x
        assert!(
            (intercept - (-19_999_999.0)).abs() / 19_999_999.0 < 1e-8,
            "intercept={intercept}"
        );
        assert!((slope - 2.0).abs() < 1e-8, "slope={slope}");
        assert!((ill.r_squared().values()[idx] - 1.0).abs() < 1e-10);
        assert!((ill.fitted_values().values()[idx] - y_ill.values()[idx]).abs() < 1e-7);
        assert!(ill.standard_deviation_of_errors().values()[idx] < 1e-7);
    }

    // 5. Out-of-range coefficient(i) returns None
    assert!(perf.coefficient(3).is_none());
    assert!(perf.coefficient(99).is_none());
    let noint = RollingLinearRegression::fit(&[x.clone(), x2], &y_perf, 4, false).unwrap();
    assert!(noint.coefficient(0).is_none());
    assert!(noint.coefficient(1).is_some());
    assert!(noint.coefficient(2).is_some());
    assert!(noint.coefficient(3).is_none());

    // 6. Bitwise determinism across back-to-back calls
    let run1 = RollingLinearRegression::fit(std::slice::from_ref(&x_ill), &y_ill, 4, true).unwrap();
    let run2 = RollingLinearRegression::fit(&[x_ill], &y_ill, 4, true).unwrap();
    assert_eq!(run1.column_indices(), run2.column_indices());
    for &col in run1.column_indices() {
        let a = run1.coefficient(col).unwrap();
        let b = run2.coefficient(col).unwrap();
        assert_eq!(a.dates(), b.dates());
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(u, v)| u.to_bits() == v.to_bits()));
    }
    for (a, b) in [
        (run1.r_squared(), run2.r_squared()),
        (run1.fitted_values(), run2.fitted_values()),
        (
            run1.standard_deviation_of_errors(),
            run2.standard_deviation_of_errors(),
        ),
    ] {
        assert_eq!(a.dates(), b.dates());
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(u, v)| u.to_bits() == v.to_bits()));
    }
}

#[test]
fn calendar_has_feb_29_and_day_count_fraction() {
    // Date accessors
    let d_leap = date(2020, 2, 29);
    assert_eq!(d_leap.year(), 2020);
    assert_eq!(d_leap.month(), 2);
    assert_eq!(d_leap.day(), 29);

    // has_feb_29: start exclusive, end inclusive (upstream test_has_feb_29 + century rules)
    assert!(has_feb_29(date(2020, 1, 1), date(2020, 3, 1)));
    assert!(has_feb_29(date(2020, 2, 28), date(2020, 2, 29)));
    assert!(!has_feb_29(date(2020, 2, 29), date(2020, 3, 1)));
    assert!(!has_feb_29(date(2020, 2, 29), date(2020, 2, 29)));
    assert!(!has_feb_29(date(2020, 3, 1), date(2020, 2, 28)));
    assert!(!has_feb_29(date(2019, 1, 1), date(2019, 12, 31)));
    assert!(!has_feb_29(date(2019, 3, 1), date(2020, 2, 28)));
    assert!(has_feb_29(date(2019, 3, 1), date(2020, 2, 29)));
    assert!(!has_feb_29(date(2020, 2, 29), date(2021, 1, 1)));

    // Century leap/non-leap rules: 1900/2100 non-leap, 2000/2400 leap
    assert!(!has_feb_29(date(1900, 1, 1), date(1900, 12, 31)));
    assert!(has_feb_29(date(2000, 1, 1), date(2000, 12, 31)));
    assert!(!has_feb_29(date(2100, 1, 1), date(2100, 12, 31)));
    assert!(has_feb_29(date(2400, 1, 1), date(2400, 12, 31)));
    assert!(!has_feb_29(date(1896, 2, 29), date(1904, 2, 28)));
    assert!(has_feb_29(date(1, 1, 1), date(9999, 12, 31)));

    // PaymentFrequency discriminants
    assert_eq!(PaymentFrequency::Daily as u32, 252);
    assert_eq!(PaymentFrequency::Weekly as u32, 52);
    assert_eq!(PaymentFrequency::SemiMonthly as u32, 26);
    assert_eq!(PaymentFrequency::Monthly as u32, 12);
    assert_eq!(PaymentFrequency::SemiQuarterly as u32, 6);
    assert_eq!(PaymentFrequency::Quarterly as u32, 4);
    assert_eq!(PaymentFrequency::TriAnnually as u32, 3);
    assert_eq!(PaymentFrequency::SemiAnnually as u32, 2);
    assert_eq!(PaymentFrequency::Annually as u32, 1);

    // day_count_fraction: all 6 conventions on 182-day leap-crossing span
    let s = date(2020, 1, 1);
    let e = date(2020, 7, 1);
    assert!(
        (day_count_fraction(
            s,
            e,
            DayCountConvention::Actual360,
            PaymentFrequency::Annually
        ) - 182.0 / 360.0)
            .abs()
            < 1e-12
    );
    assert!(
        (day_count_fraction(
            s,
            e,
            DayCountConvention::Actual364,
            PaymentFrequency::Annually
        ) - 182.0 / 364.0)
            .abs()
            < 1e-12
    );
    assert!(
        (day_count_fraction(
            s,
            e,
            DayCountConvention::Actual36525,
            PaymentFrequency::Annually
        ) - 182.0 / 365.25)
            .abs()
            < 1e-12
    );
    assert!(
        (day_count_fraction(
            s,
            e,
            DayCountConvention::Actual365F,
            PaymentFrequency::Annually
        ) - 182.0 / 365.0)
            .abs()
            < 1e-12
    );
    assert!(
        (day_count_fraction(
            s,
            e,
            DayCountConvention::Actual365L,
            PaymentFrequency::Annually
        ) - 182.0 / 366.0)
            .abs()
            < 1e-12
    );
    assert_eq!(
        day_count_fraction(s, e, DayCountConvention::OneOne, PaymentFrequency::Annually),
        1.0
    );

    // Actual365L: Annually checks has_feb_29(start, end); non-Annually checks is_leap(end.year)
    let mar1 = date(2020, 3, 1);
    let sep1 = date(2020, 9, 1);
    assert!(
        (day_count_fraction(
            mar1,
            sep1,
            DayCountConvention::Actual365L,
            PaymentFrequency::Annually
        ) - 184.0 / 365.0)
            .abs()
            < 1e-12
    );
    for freq in [
        PaymentFrequency::Daily,
        PaymentFrequency::Weekly,
        PaymentFrequency::SemiMonthly,
        PaymentFrequency::Monthly,
        PaymentFrequency::SemiQuarterly,
        PaymentFrequency::Quarterly,
        PaymentFrequency::TriAnnually,
        PaymentFrequency::SemiAnnually,
    ] {
        assert!(
            (day_count_fraction(mar1, sep1, DayCountConvention::Actual365L, freq) - 184.0 / 366.0)
                .abs()
                < 1e-12
        );
    }
    // Crosses Feb 29 2020, ends in non-leap 2021
    assert!(
        (day_count_fraction(
            date(2020, 1, 1),
            date(2021, 1, 1),
            DayCountConvention::Actual365L,
            PaymentFrequency::Annually
        ) - 1.0)
            .abs()
            < 1e-12
    );
    assert!(
        (day_count_fraction(
            date(2020, 1, 1),
            date(2021, 1, 1),
            DayCountConvention::Actual365L,
            PaymentFrequency::Monthly
        ) - 366.0 / 365.0)
            .abs()
            < 1e-12
    );

    // Zero & negative spans
    assert_eq!(
        day_count_fraction(
            s,
            s,
            DayCountConvention::Actual360,
            PaymentFrequency::Annually
        ),
        0.0
    );
    assert_eq!(
        day_count_fraction(s, s, DayCountConvention::OneOne, PaymentFrequency::Annually),
        1.0
    );
    assert!(
        (day_count_fraction(
            date(2020, 3, 1),
            date(2020, 1, 1),
            DayCountConvention::Actual365L,
            PaymentFrequency::Annually
        ) - (-60.0 / 365.0))
            .abs()
            < 1e-12
    );
    assert!(
        (day_count_fraction(
            date(2020, 3, 1),
            date(2020, 1, 1),
            DayCountConvention::Actual365L,
            PaymentFrequency::Monthly
        ) - (-60.0 / 366.0))
            .abs()
            < 1e-12
    );
    assert_eq!(
        day_count_fraction(e, s, DayCountConvention::OneOne, PaymentFrequency::Annually),
        1.0
    );
}

#[test]
fn calendar_holiday_construction_and_is_business_day() {
    // All-false week_mask -> Error::Date
    assert_eq!(
        HolidayCalendar::try_new([false; 7], vec![]),
        Err(Error::Date)
    );
    assert_eq!(
        HolidayCalendar::try_new([false; 7], vec![date(2020, 1, 1)]),
        Err(Error::Date)
    );

    // Default Mon-Fri weekdays with unsorted, duplicate, and weekend holidays normalized
    let cal = HolidayCalendar::weekdays(vec![
        date(2020, 1, 4), // Sat -> filtered out
        date(2020, 1, 5), // Sun -> filtered out
        date(2020, 1, 6), // Mon
        date(2020, 1, 1), // Wed
        date(2020, 1, 1), // duplicate Wed
    ]);
    assert_eq!(
        cal.week_mask(),
        [true, true, true, true, true, false, false]
    );
    assert_eq!(cal.holidays(), &[date(2020, 1, 1), date(2020, 1, 6)]);

    assert!(!is_business_day(date(2020, 1, 1), &cal)); // holiday
    assert!(is_business_day(date(2020, 1, 2), &cal)); // Thu
    assert!(is_business_day(date(2020, 1, 3), &cal)); // Fri
    assert!(!is_business_day(date(2020, 1, 4), &cal)); // Sat
    assert!(!is_business_day(date(2020, 1, 5), &cal)); // Sun
    assert!(!is_business_day(date(2020, 1, 6), &cal)); // holiday
    assert!(is_business_day(date(2020, 1, 7), &cal)); // Tue
    assert!(!is_business_day(date(2020, 2, 29), &cal)); // Sat leap day
    assert!(is_business_day(date(2024, 2, 29), &cal)); // Thu leap day
    assert!(is_business_day(date(1, 1, 1), &cal)); // Mon
    assert!(is_business_day(date(9999, 12, 31), &cal)); // Fri

    // Custom Sun-Thu weekmask (Mon..Thu + Sun)
    let sun_thu_mask = [true, true, true, true, false, false, true];
    let sun_thu = HolidayCalendar::try_new(
        sun_thu_mask,
        vec![
            date(2020, 1, 3), // Fri -> filtered out
            date(2020, 1, 4), // Sat -> filtered out
            date(2020, 1, 5), // Sun -> kept
            date(2020, 1, 5), // duplicate Sun
        ],
    )
    .unwrap();
    assert_eq!(sun_thu.week_mask(), sun_thu_mask);
    assert_eq!(sun_thu.holidays(), &[date(2020, 1, 5)]);
    assert!(!is_business_day(date(2020, 1, 3), &sun_thu)); // Fri off
    assert!(!is_business_day(date(2020, 1, 4), &sun_thu)); // Sat off
    assert!(!is_business_day(date(2020, 1, 5), &sun_thu)); // Sun holiday
    assert!(is_business_day(date(2020, 1, 12), &sun_thu)); // Sun business day
}

#[test]
fn calendar_business_day_offset_rolls_large_and_bounds() {
    let cal_ny = HolidayCalendar::weekdays(vec![date(2020, 1, 1)]);

    // Business day with Roll::Raise
    assert_eq!(
        business_day_offset(date(2020, 1, 2), 0, Roll::Raise, &cal_ny),
        Ok(date(2020, 1, 2))
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 2), 2, Roll::Raise, &cal_ny),
        Ok(date(2020, 1, 6))
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 2), -1, Roll::Raise, &cal_ny),
        Ok(date(2019, 12, 31))
    );
    // Non-business day with Roll::Raise -> Error::Date
    assert_eq!(
        business_day_offset(date(2020, 1, 1), 0, Roll::Raise, &cal_ny),
        Err(Error::Date)
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 4), 1, Roll::Raise, &cal_ny),
        Err(Error::Date)
    );

    // Month-end holiday 2020-01-31 (Fri): all 4 non-raising rolls
    let cal_jan31 = HolidayCalendar::weekdays(vec![date(2020, 1, 31)]);
    assert_eq!(
        business_day_offset(date(2020, 1, 31), 0, Roll::Forward, &cal_jan31),
        Ok(date(2020, 2, 3))
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 31), 0, Roll::Backward, &cal_jan31),
        Ok(date(2020, 1, 30))
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 31), 0, Roll::ModifiedFollowing, &cal_jan31),
        Ok(date(2020, 1, 30))
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 31), 1, Roll::ModifiedFollowing, &cal_jan31),
        Ok(date(2020, 2, 3))
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 31), 0, Roll::ModifiedPreceding, &cal_jan31),
        Ok(date(2020, 1, 30))
    );

    // Month-start weekend + holiday 2020-02-03 (Mon)
    let cal_feb3 = HolidayCalendar::weekdays(vec![date(2020, 2, 3)]);
    assert_eq!(
        business_day_offset(date(2020, 2, 1), 0, Roll::Backward, &cal_feb3),
        Ok(date(2020, 1, 31))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 1), 0, Roll::ModifiedPreceding, &cal_feb3),
        Ok(date(2020, 2, 4))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 3), 0, Roll::ModifiedPreceding, &cal_feb3),
        Ok(date(2020, 2, 4))
    );

    // Leap day 2020-02-29 (Sat)
    let empty_cal = HolidayCalendar::weekdays(vec![]);
    assert_eq!(
        business_day_offset(date(2020, 2, 29), 0, Roll::Forward, &empty_cal),
        Ok(date(2020, 3, 2))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 29), 0, Roll::Backward, &empty_cal),
        Ok(date(2020, 2, 28))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 29), 0, Roll::ModifiedFollowing, &empty_cal),
        Ok(date(2020, 2, 28))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 29), 0, Roll::ModifiedPreceding, &empty_cal),
        Ok(date(2020, 2, 28))
    );

    // All-month-holiday Feb 2020 fallbacks
    let all_feb_2020: Vec<Date> = (1..=29).map(|d| date(2020, 2, d)).collect();
    let cal_all_feb = HolidayCalendar::weekdays(all_feb_2020);
    assert_eq!(
        business_day_offset(date(2020, 2, 15), 0, Roll::ModifiedFollowing, &cal_all_feb),
        Ok(date(2020, 1, 31))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 15), 1, Roll::ModifiedFollowing, &cal_all_feb),
        Ok(date(2020, 3, 2))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 15), 0, Roll::ModifiedPreceding, &cal_all_feb),
        Ok(date(2020, 3, 2))
    );
    assert_eq!(
        business_day_offset(date(2020, 2, 15), -1, Roll::ModifiedPreceding, &cal_all_feb),
        Ok(date(2020, 1, 31))
    );

    // Large offsets (+/-5000 business days) & round-trip consistency across centuries
    let start = date(2000, 1, 3);
    let fwd = business_day_offset(start, 5000, Roll::Raise, &cal_ny).unwrap();
    let back = business_day_offset(fwd, -5000, Roll::Raise, &cal_ny).unwrap();
    assert_eq!(back, start);
    assert_eq!(business_day_count(start, fwd, &cal_ny), 5000);
    assert_eq!(business_day_count(fwd, start, &cal_ny), -5000);

    // Extreme i32::MIN / i32::MAX do not overflow/panic
    assert_eq!(
        business_day_offset(date(2020, 1, 2), i32::MIN, Roll::Raise, &empty_cal),
        Err(Error::Date)
    );
    assert_eq!(
        business_day_offset(date(2020, 1, 2), i32::MAX, Roll::Raise, &empty_cal),
        Err(Error::Date)
    );

    // Gregorian 0001-01-01 / 9999-12-31 bounds
    assert_eq!(
        business_day_offset(date(1, 1, 1), -1, Roll::Backward, &empty_cal),
        Err(Error::Date)
    );
    assert_eq!(
        business_day_offset(date(9999, 12, 31), 1, Roll::Forward, &empty_cal),
        Err(Error::Date)
    );
    let cal_edge_hols = HolidayCalendar::weekdays(vec![date(1, 1, 1), date(9999, 12, 31)]);
    assert_eq!(
        business_day_offset(date(1, 1, 1), 0, Roll::Backward, &cal_edge_hols),
        Err(Error::Date)
    );
    assert_eq!(
        business_day_offset(date(1, 1, 1), 0, Roll::ModifiedPreceding, &cal_edge_hols),
        Ok(date(1, 1, 2))
    );
    assert_eq!(
        business_day_offset(date(9999, 12, 31), 0, Roll::Forward, &cal_edge_hols),
        Err(Error::Date)
    );
    assert_eq!(
        business_day_offset(
            date(9999, 12, 31),
            0,
            Roll::ModifiedFollowing,
            &cal_edge_hols
        ),
        Ok(date(9999, 12, 30))
    );
}

#[test]
fn calendar_business_day_count_asymmetry_spans_and_determinism() {
    let cal_ny = HolidayCalendar::weekdays(vec![date(2020, 1, 1)]);

    // Exact NumPy asymmetry: [begin, end) forward vs -(end, begin] reversed
    assert_eq!(
        business_day_count(date(2020, 1, 1), date(2020, 1, 6), &cal_ny),
        2
    );
    assert_eq!(
        business_day_count(date(2020, 1, 6), date(2020, 1, 1), &cal_ny),
        -3
    );
    assert_eq!(
        business_day_count(date(2020, 1, 4), date(2020, 1, 6), &cal_ny),
        0
    );
    assert_eq!(
        business_day_count(date(2020, 1, 6), date(2020, 1, 4), &cal_ny),
        -1
    );
    assert_eq!(
        business_day_count(date(2020, 1, 2), date(2020, 1, 6), &cal_ny),
        2
    );
    assert_eq!(
        business_day_count(date(2020, 1, 6), date(2020, 1, 2), &cal_ny),
        -2
    );
    assert_eq!(
        business_day_count(date(2020, 1, 6), date(2020, 1, 6), &cal_ny),
        0
    );
    assert_eq!(
        business_day_count(date(2020, 1, 1), date(2020, 1, 1), &cal_ny),
        0
    );

    // Multi-decade span & offset round-trip consistency
    let hols = vec![
        date(1980, 1, 1),
        date(2000, 2, 29),
        date(2020, 1, 1),
        date(2024, 2, 29),
        date(2040, 7, 4),
    ];
    let cal = HolidayCalendar::weekdays(hols);
    let b1 = date(1970, 1, 1);
    let e1 = date(2050, 1, 3);
    let cnt_fwd = business_day_count(b1, e1, &cal);
    let cnt_rev = business_day_count(e1, b1, &cal);
    assert_eq!(cnt_fwd, -cnt_rev); // both b1 (Thu) and e1 (Mon) are business days
    assert_eq!(
        business_day_offset(b1, cnt_fwd as i32, Roll::Raise, &cal),
        Ok(e1)
    );

    // Full Gregorian range 0001-01-01..9999-12-31
    let empty_cal = HolidayCalendar::weekdays(vec![]);
    let full_fwd = business_day_count(date(1, 1, 1), date(9999, 12, 31), &empty_cal);
    let full_rev = business_day_count(date(9999, 12, 31), date(1, 1, 1), &empty_cal);
    assert_eq!(full_fwd, -full_rev);
    assert_eq!(
        business_day_offset(date(1, 1, 1), full_fwd as i32, Roll::Raise, &empty_cal),
        Ok(date(9999, 12, 31))
    );

    // Custom weekmask (Sun-Thu) and determinism across back-to-back calls
    let sun_thu = HolidayCalendar::try_new(
        [true, true, true, true, false, false, true],
        vec![date(2020, 1, 5)],
    )
    .unwrap();
    let run1 = business_day_count(date(2020, 1, 1), date(2020, 1, 15), &sun_thu);
    let run2 = business_day_count(date(2020, 1, 1), date(2020, 1, 15), &sun_thu);
    assert_eq!(run1, run2);
    assert_eq!(run1, 9);
}

#[test]
fn basket_upstream_scenarios_and_basket_series_equivalence() {
    let d5: Vec<Date> = (1..=5).map(|d| date(2024, 1, d)).collect();
    let s1 = Series::try_new(d5.clone(), vec![100.0, 110.0, 99.0, 108.9, 120.0]).unwrap();
    let s2 = Series::try_new(d5.clone(), vec![50.0, 50.0, 55.0, 52.25, 60.0]).unwrap();
    let s3 = Series::try_new(d5.clone(), vec![200.0, 220.0, 210.0, 231.0, 240.0]).unwrap();

    // 1 asset default weights: NAV equals 100 * p[i] / p[0], actual weight stays 1.0
    let res1 = backtest_basket(std::slice::from_ref(&s1), None, None, RebalFreq::Daily).unwrap();
    assert_eq!(res1.nav.dates(), d5.as_slice());
    assert_eq!(res1.actual_weights.len(), 1);
    for (i, &p) in s1.values().iter().enumerate() {
        assert!((res1.nav.values()[i] - 100.0 * p / s1.values()[0]).abs() < 1e-12);
        assert!((res1.actual_weights[0].values()[i] - 1.0).abs() < 1e-12);
    }

    // [1.0, 0.0] and [0.0, 1.0] select asset 0 and asset 1 respectively
    let w10 = [BasketParam::Fixed(1.0), BasketParam::Fixed(0.0)];
    let w01 = [BasketParam::Fixed(0.0), BasketParam::Fixed(1.0)];
    let r10 = backtest_basket(
        &[s1.clone(), s2.clone()],
        Some(&w10),
        None,
        RebalFreq::Daily,
    )
    .unwrap();
    let r01 = backtest_basket(
        &[s1.clone(), s2.clone()],
        Some(&w01),
        None,
        RebalFreq::Daily,
    )
    .unwrap();
    for i in 0..d5.len() {
        assert!((r10.nav.values()[i] - 100.0 * s1.values()[i] / s1.values()[0]).abs() < 1e-12);
        assert!((r01.nav.values()[i] - 100.0 * s2.values()[i] / s2.values()[0]).abs() < 1e-12);
        assert_eq!(r10.actual_weights[0].values()[i], 1.0);
        assert_eq!(r10.actual_weights[1].values()[i], 0.0);
        assert_eq!(r01.actual_weights[0].values()[i], 0.0);
        assert_eq!(r01.actual_weights[1].values()[i], 1.0);
    }

    // 2-asset and 3-asset equal-weight daily baskets
    let r2 = backtest_basket(&[s1.clone(), s2.clone()], None, None, RebalFreq::Daily).unwrap();
    // Day 1: 0.5*(+10%) + 0.5*(0%) = +5% -> 105.0
    assert!((r2.nav.values()[1] - 105.0).abs() < 1e-12);
    // Day 2: 0.5*(-10%) + 0.5*(+10%) = 0% -> 105.0
    assert!((r2.nav.values()[2] - 105.0).abs() < 1e-12);
    let r3 = backtest_basket(
        &[s1.clone(), s2.clone(), s3.clone()],
        None,
        None,
        RebalFreq::Daily,
    )
    .unwrap();
    // Day 1: (10% + 0% + 10%) / 3 = +20/3 % -> 100 + 20/3
    assert!((r3.nav.values()[1] - (100.0 + 20.0 / 3.0)).abs() < 1e-12);
    assert_eq!(r3.actual_weights.len(), 3);

    // Monthly single-asset basket over multiple months equals normalized price series
    let monthly_dates = vec![
        date(2024, 1, 10),
        date(2024, 2, 12),
        date(2024, 3, 15),
        date(2024, 4, 20),
    ];
    let sm = Series::try_new(monthly_dates.clone(), vec![80.0, 88.0, 96.0, 100.0]).unwrap();
    let rm = backtest_basket(std::slice::from_ref(&sm), None, None, RebalFreq::Monthly).unwrap();
    for (i, &p) in sm.values().iter().enumerate() {
        assert!((rm.nav.values()[i] - 100.0 * p / 80.0).abs() < 1e-12);
    }

    // Weekly leveraged [1.0, 1.0] basket
    let wk_dates = vec![
        date(2024, 1, 1),
        date(2024, 1, 3),
        date(2024, 1, 8),
        date(2024, 1, 12),
    ];
    let wa = Series::try_new(wk_dates.clone(), vec![100.0, 110.0, 120.0, 126.0]).unwrap();
    let wb = Series::try_new(wk_dates, vec![100.0, 105.0, 100.0, 110.0]).unwrap();
    let w11 = [BasketParam::Fixed(1.0), BasketParam::Fixed(1.0)];
    let rw = backtest_basket(
        &[wa.clone(), wb.clone()],
        Some(&w11),
        None,
        RebalFreq::Weekly,
    )
    .unwrap();
    // Day 1 (Jan 3): +10 + 5 = 115.0; Day 2 (Jan 8, rebal): +10 - 5 = 120.0, reset units to 120/120=1, 120/100=1.2
    // Day 3 (Jan 12): 120 + 1*(6) + 1.2*(10) = 138.0
    assert!((rw.nav.values()[1] - 115.0).abs() < 1e-12);
    assert!((rw.nav.values()[2] - 120.0).abs() < 1e-12);
    assert!((rw.nav.values()[3] - 138.0).abs() < 1e-12);

    // basket_series is bit-exact with backtest_basket(...).nav
    let costs = [BasketParam::Fixed(0.001), BasketParam::Fixed(0.002)];
    let full = backtest_basket(
        &[wa.clone(), wb.clone()],
        Some(&w11),
        Some(&costs),
        RebalFreq::Weekly,
    )
    .unwrap();
    let nav_only = basket_series(&[wa, wb], Some(&w11), Some(&costs), RebalFreq::Weekly).unwrap();
    assert_eq!(full.nav.dates(), nav_only.dates());
    for (&a, &b) in full.nav.values().iter().zip(nav_only.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn basket_rebalance_schedules_strict_max_gap_snap_and_jan31() {
    let w50 = [BasketParam::Fixed(0.5), BasketParam::Fixed(0.5)];
    let c10 = [BasketParam::Fixed(0.1), BasketParam::Fixed(0.1)];

    // 1. Exact 7-day weekly span: hyp = 2024-01-08 == max(cal) -> NOT rebalanced on last day!
    let d_exact7 = vec![date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 8)];
    let p_a7 = Series::try_new(d_exact7.clone(), vec![100.0, 120.0, 150.0]).unwrap();
    let p_b7 = Series::try_new(d_exact7, vec![100.0, 100.0, 100.0]).unwrap();
    let r_exact7 =
        backtest_basket(&[p_a7, p_b7], Some(&w50), Some(&c10), RebalFreq::Weekly).unwrap();
    // No rebalance on Jan 8 -> NAV = 100 + 0.5*50 + 0.5*0 = 125.0 (zero cost charged!), aw = [0.6, 0.4]
    assert!((r_exact7.nav.values()[2] - 125.0).abs() < 1e-12);
    assert!((r_exact7.actual_weights[0].values()[2] - 0.6).abs() < 1e-12);
    assert!((r_exact7.actual_weights[1].values()[2] - 0.4).abs() < 1e-12);

    // 2. Holiday-gap snap onto cal[n-1]: hyp = 2024-01-08 < 2024-01-10 snaps to Jan 10 -> IS rebalanced!
    let d_snap = vec![date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 10)];
    let p_as = Series::try_new(d_snap.clone(), vec![100.0, 120.0, 150.0]).unwrap();
    let p_bs = Series::try_new(d_snap, vec![100.0, 100.0, 100.0]).unwrap();
    let r_snap = backtest_basket(&[p_as, p_bs], Some(&w50), Some(&c10), RebalFreq::Weekly).unwrap();
    // Pre-cost NAV = 125.0, aw = [0.6, 0.4], cost_rate = 0.1*0.1 + 0.1*0.1 = 0.02 -> NAV = 122.5, aw reset to [0.5, 0.5]
    assert!((r_snap.nav.values()[2] - 122.5).abs() < 1e-12);
    assert_eq!(r_snap.actual_weights[0].values()[2], 0.5);
    assert_eq!(r_snap.actual_weights[1].values()[2], 0.5);

    // 3. Multiple hypothetical weekly dates (Jan 8, Jan 15, Jan 22) snapping into the same gap rebalance ONCE
    let d_dedup = vec![
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 25),
        date(2024, 1, 26),
    ];
    let p_ad = Series::try_new(d_dedup.clone(), vec![100.0, 120.0, 150.0, 150.0]).unwrap();
    let p_bd = Series::try_new(d_dedup, vec![100.0, 100.0, 100.0, 100.0]).unwrap();
    let r_dedup =
        backtest_basket(&[p_ad, p_bd], Some(&w50), Some(&c10), RebalFreq::Weekly).unwrap();
    assert!((r_dedup.nav.values()[2] - 122.5).abs() < 1e-12);
    assert!((r_dedup.nav.values()[3] - 122.5).abs() < 1e-12);

    // 4. Monthly Jan 31 leap anchor (2024): 2024-01-31 -> 2024-02-29 -> 2024-03-31 -> 2024-04-30
    let d_leap = vec![
        date(2024, 1, 31),
        date(2024, 2, 28), // NOT rebal (Feb 29 is leap day)
        date(2024, 2, 29), // REBAL (k=1)
        date(2024, 3, 29), // NOT rebal (anchor is 31, not chained 29!)
        date(2024, 3, 31), // REBAL (k=2)
        date(2024, 4, 29), // NOT rebal
        date(2024, 4, 30), // REBAL (k=3, < May 1)
        date(2024, 5, 1),  // NOT rebal
    ];
    let p_al = Series::try_new(
        d_leap.clone(),
        vec![100.0, 120.0, 150.0, 180.0, 200.0, 220.0, 250.0, 260.0],
    )
    .unwrap();
    let p_bl = Series::try_new(d_leap, vec![100.0; 8]).unwrap();
    let r_leap =
        backtest_basket(&[p_al, p_bl], Some(&w50), Some(&c10), RebalFreq::Monthly).unwrap();
    // Non-rebal dates drift away from 0.5; rebal dates reset to exact 0.5
    assert!((r_leap.actual_weights[0].values()[1] - 0.5).abs() > 1e-3); // Feb 28 drifts
    assert_eq!(r_leap.actual_weights[0].values()[2], 0.5); // Feb 29 rebalances
    assert!((r_leap.actual_weights[0].values()[3] - 0.5).abs() > 1e-3); // Mar 29 drifts!
    assert_eq!(r_leap.actual_weights[0].values()[4], 0.5); // Mar 31 rebalances
    assert!((r_leap.actual_weights[0].values()[5] - 0.5).abs() > 1e-3); // Apr 29 drifts
    assert_eq!(r_leap.actual_weights[0].values()[6], 0.5); // Apr 30 rebalances
    assert!((r_leap.actual_weights[0].values()[7] - 0.5).abs() > 1e-3); // May 1 drifts

    // 5. Monthly Jan 31 non-leap anchor (2023): 2023-01-31 -> 2023-02-28 -> 2023-03-31 -> 2023-04-30
    let d_nonleap = vec![
        date(2023, 1, 31),
        date(2023, 2, 27), // drifts
        date(2023, 2, 28), // REBAL (k=1)
        date(2023, 3, 28), // drifts (proves anchor is Jan 31, NOT chained Feb 28!)
        date(2023, 3, 31), // REBAL (k=2)
        date(2023, 4, 30), // NOT rebal (hyp == max(cal) excluded!)
    ];
    let p_an = Series::try_new(
        d_nonleap.clone(),
        vec![100.0, 120.0, 150.0, 180.0, 200.0, 240.0],
    )
    .unwrap();
    let p_bn = Series::try_new(d_nonleap, vec![100.0; 6]).unwrap();
    let r_nonleap =
        backtest_basket(&[p_an, p_bn], Some(&w50), Some(&c10), RebalFreq::Monthly).unwrap();
    assert!((r_nonleap.actual_weights[0].values()[1] - 0.5).abs() > 1e-3);
    assert_eq!(r_nonleap.actual_weights[0].values()[2], 0.5);
    assert!((r_nonleap.actual_weights[0].values()[3] - 0.5).abs() > 1e-3);
    assert_eq!(r_nonleap.actual_weights[0].values()[4], 0.5);
    assert!((r_nonleap.actual_weights[0].values()[5] - 0.5).abs() > 1e-3); // Apr 30 == max(cal) excluded
}

#[test]
fn basket_series_params_calendar_alignment_leverage_zero_weights_and_singleton() {
    // 1. Misaligned price + weight Series + cost Series intersection
    let p1 = Series::try_new(
        vec![
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
        ],
        vec![90.0, 100.0, 110.0, 120.0, 130.0],
    )
    .unwrap();
    let p2 = Series::try_new(
        vec![
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 6),
        ],
        vec![50.0, 50.0, 55.0, 60.0],
    )
    .unwrap();
    let w1_s = Series::try_new(
        vec![date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        vec![0.6, 0.8, 0.5],
    )
    .unwrap();
    let c2_s = Series::try_new(
        vec![date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        vec![0.001, 0.002, 0.003],
    )
    .unwrap();
    let weights = [BasketParam::Series(&w1_s), BasketParam::Fixed(0.4)];
    let costs = [BasketParam::Fixed(0.001), BasketParam::Series(&c2_s)];
    let res = backtest_basket(&[p1, p2], Some(&weights), Some(&costs), RebalFreq::Daily).unwrap();
    assert_eq!(
        res.nav.dates(),
        &[date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    );
    // Day 0 (Jan 2): nav = 100.0, aw = [0.6, 0.4]
    // Day 1 (Jan 3): p1: 100->110 (+10%), p2: 50->50 (0%) -> pre-cost nav = 106.0
    //   aw_pre = [0.6*1.1*(100/106), 0.4*1.0*(100/106)] = [66/106, 40/106]
    //   cost_rate = 0.001*|0.8 - 66/106| + 0.002*|0.4 - 40/106|
    let aw0_pre: f64 = 66.0 / 106.0;
    let aw1_pre: f64 = 40.0 / 106.0;
    let expected_nav1 =
        106.0 - (0.001 * (0.8 - aw0_pre).abs() + 0.002 * (0.4 - aw1_pre).abs()) * 106.0;
    assert!((res.nav.values()[1] - expected_nav1).abs() < 1e-12);
    assert_eq!(res.actual_weights[0].values()[1], 0.8);
    assert_eq!(res.actual_weights[1].values()[1], 0.4);

    // 2. Short / leveraged weights (sum != 1 and net short)
    let d3 = vec![date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 5)];
    let pa = Series::try_new(d3.clone(), vec![100.0, 110.0, 121.0]).unwrap();
    let pb = Series::try_new(d3.clone(), vec![100.0, 90.0, 99.0]).unwrap();
    let w_ls = [BasketParam::Fixed(1.5), BasketParam::Fixed(-0.5)];
    let r_ls = backtest_basket(
        &[pa.clone(), pb.clone()],
        Some(&w_ls),
        None,
        RebalFreq::Daily,
    )
    .unwrap();
    // Day 1: 1.5*(+10%) + (-0.5)*(-10%) = +20% -> 120.0
    assert!((r_ls.nav.values()[1] - 120.0).abs() < 1e-12);

    // 3. All-zero weights: NAV stays 100.0, actual weights stay 0.0
    let w_zero = [BasketParam::Fixed(0.0), BasketParam::Fixed(0.0)];
    let r_zero = backtest_basket(
        &[pa.clone(), pb.clone()],
        Some(&w_zero),
        None,
        RebalFreq::Daily,
    )
    .unwrap();
    assert_eq!(r_zero.nav.values(), &[100.0, 100.0, 100.0]);
    assert_eq!(r_zero.actual_weights[0].values(), &[0.0, 0.0, 0.0]);
    assert_eq!(r_zero.actual_weights[1].values(), &[0.0, 0.0, 0.0]);

    // 4. Single-date series across Daily, Weekly, Monthly
    let s1 = Series::try_new(vec![date(2024, 3, 1)], vec![123.45]).unwrap();
    let s2 = Series::try_new(vec![date(2024, 3, 1)], vec![67.89]).unwrap();
    let w73 = [BasketParam::Fixed(0.7), BasketParam::Fixed(0.3)];
    for freq in [RebalFreq::Daily, RebalFreq::Weekly, RebalFreq::Monthly] {
        let r = backtest_basket(&[s1.clone(), s2.clone()], Some(&w73), None, freq).unwrap();
        assert_eq!(r.nav.dates(), &[date(2024, 3, 1)]);
        assert_eq!(r.nav.values(), &[100.0]);
        assert_eq!(r.actual_weights[0].values(), &[0.7]);
        assert_eq!(r.actual_weights[1].values(), &[0.3]);
    }
}

#[test]
fn basket_nonfinite_zero_negative_nav_errors_and_determinism() {
    let d3 = vec![date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 3)];

    // 1. NaN price propagates cleanly through NAV and actual weights
    let p_nan = Series::try_new(d3.clone(), vec![100.0, f64::NAN, 110.0]).unwrap();
    let p_ok = Series::try_new(d3.clone(), vec![100.0, 105.0, 110.0]).unwrap();
    let r_nan = backtest_basket(&[p_nan, p_ok.clone()], None, None, RebalFreq::Daily).unwrap();
    assert_eq!(r_nan.nav.values()[0], 100.0);
    assert!(r_nan.nav.values()[1].is_nan());
    assert!(r_nan.nav.values()[2].is_nan());

    // 2. Zero price on day 0 -> units = +Inf -> Weekly keeps +Inf on non-rebal day 1,
    //    Daily rebalances on day 1 where 0.0 * |w - NaN| propagates NaN into NAV.
    let p_z0 = Series::try_new(d3.clone(), vec![0.0, 10.0, 20.0]).unwrap();
    let r_z0_w =
        backtest_basket(&[p_z0.clone(), p_ok.clone()], None, None, RebalFreq::Weekly).unwrap();
    assert_eq!(r_z0_w.nav.values()[0], 100.0);
    assert!(r_z0_w.nav.values()[1].is_infinite() && r_z0_w.nav.values()[1].is_sign_positive());
    assert!(r_z0_w.actual_weights[0].values()[1].is_nan());
    let r_z0_d = backtest_basket(&[p_z0, p_ok.clone()], None, None, RebalFreq::Daily).unwrap();
    assert_eq!(r_z0_d.nav.values()[0], 100.0);
    assert!(r_z0_d.nav.values()[1].is_nan());

    // 3. Zero NAV (100 -> 0 -> 0) propagates IEEE-754 cleanly
    let p_crash = Series::try_new(d3.clone(), vec![100.0, 0.0, 10.0]).unwrap();
    let w1 = [BasketParam::Fixed(1.0)];
    let r_znav = backtest_basket(
        std::slice::from_ref(&p_crash),
        Some(&w1),
        None,
        RebalFreq::Daily,
    )
    .unwrap();
    assert_eq!(r_znav.nav.values()[0], 100.0);
    assert!(r_znav.nav.values()[1].is_nan()); // aw = 1.0 * 0.0 * (100/0) = NaN, cost = 0*NaN = NaN -> nav_i = NaN

    // 4. Negative NAV with leveraged weight & cost sign (nav_i -= cost_rate * nav_i)
    let p_drop = Series::try_new(d3.clone(), vec![100.0, 40.0, 50.0]).unwrap();
    let w2 = [BasketParam::Fixed(2.0)];
    let c01 = [BasketParam::Fixed(0.01)];
    let r_neg = backtest_basket(
        std::slice::from_ref(&p_drop),
        Some(&w2),
        Some(&c01),
        RebalFreq::Daily,
    )
    .unwrap();
    // Day 1 pre-cost: units = 2.0, pnl = 2.0 * (-60) = -120 -> nav_pre = -20.0
    // aw_pre = 2.0 * (40/100) * (100 / -20) = -4.0
    // cost_rate = 0.01 * |2.0 - (-4.0)| = 0.06
    // nav_i = -20.0 - 0.06 * (-20.0) = -18.8 (exact signed multiplication as upstream!)
    assert!((r_neg.nav.values()[1] - (-18.8)).abs() < 1e-12);

    // 5. Typed Error::Length errors
    let s_a =
        Series::try_new(vec![date(2024, 1, 1), date(2024, 1, 2)], vec![100.0, 101.0]).unwrap();
    let s_b = Series::try_new(vec![date(2024, 2, 1), date(2024, 2, 2)], vec![50.0, 51.0]).unwrap();
    let empty_s = Series::try_new(vec![], vec![]).unwrap();
    assert_eq!(
        backtest_basket(&[], None, None, RebalFreq::Daily),
        Err(Error::Length)
    );
    assert_eq!(
        basket_series(&[], None, None, RebalFreq::Daily),
        Err(Error::Length)
    );
    assert_eq!(
        backtest_basket(
            &[s_a.clone(), s_a.clone()],
            Some(&[BasketParam::Fixed(1.0)]),
            None,
            RebalFreq::Daily
        ),
        Err(Error::Length)
    );
    assert_eq!(
        backtest_basket(
            &[s_a.clone(), s_a.clone()],
            None,
            Some(&[BasketParam::Fixed(0.01)]),
            RebalFreq::Daily
        ),
        Err(Error::Length)
    );
    // Explicit Some(&[]) on non-empty series returns Err(Error::Length)
    assert_eq!(
        backtest_basket(
            std::slice::from_ref(&s_a),
            Some(&[]),
            None,
            RebalFreq::Daily
        ),
        Err(Error::Length)
    );
    assert_eq!(
        backtest_basket(
            std::slice::from_ref(&s_a),
            None,
            Some(&[]),
            RebalFreq::Daily
        ),
        Err(Error::Length)
    );
    // Disjoint calendars (prices, weight series, cost series, empty single series)
    assert_eq!(
        backtest_basket(&[s_a.clone(), s_b.clone()], None, None, RebalFreq::Daily),
        Err(Error::Length)
    );
    assert_eq!(
        backtest_basket(
            std::slice::from_ref(&s_a),
            Some(&[BasketParam::Series(&s_b)]),
            None,
            RebalFreq::Daily
        ),
        Err(Error::Length)
    );
    assert_eq!(
        backtest_basket(
            std::slice::from_ref(&s_a),
            None,
            Some(&[BasketParam::Series(&s_b)]),
            RebalFreq::Daily
        ),
        Err(Error::Length)
    );
    assert_eq!(
        backtest_basket(&[empty_s], None, None, RebalFreq::Daily),
        Err(Error::Length)
    );

    // 6. Bit-for-bit determinism across repeated calls
    let run1 = backtest_basket(
        &[p_drop.clone(), p_ok.clone()],
        None,
        Some(&[BasketParam::Fixed(0.001), BasketParam::Fixed(0.002)]),
        RebalFreq::Daily,
    )
    .unwrap();
    let run2 = backtest_basket(
        &[p_drop, p_ok],
        None,
        Some(&[BasketParam::Fixed(0.001), BasketParam::Fixed(0.002)]),
        RebalFreq::Daily,
    )
    .unwrap();
    assert_eq!(run1.nav.dates(), run2.nav.dates());
    for (&u, &v) in run1.nav.values().iter().zip(run2.nav.values()) {
        assert_eq!(u.to_bits(), v.to_bits());
    }
    for (wa, wb) in run1.actual_weights.iter().zip(&run2.actual_weights) {
        assert_eq!(wa.dates(), wb.dates());
        for (&u, &v) in wa.values().iter().zip(wb.values()) {
            assert_eq!(u.to_bits(), v.to_bits());
        }
    }
}

fn risk_row(
    d: Option<Date>,
    t: Option<&str>,
    mkt_type: &str,
    mkt_asset: &str,
    mkt_class: &str,
    mkt_point: &str,
    value: f64,
) -> RiskRow {
    RiskRow {
        date: d,
        time: t.map(str::to_string),
        mkt_type: mkt_type.to_string(),
        mkt_asset: mkt_asset.to_string(),
        mkt_class: mkt_class.to_string(),
        mkt_point: mkt_point.to_string(),
        value,
    }
}

#[test]
fn risk_point_sort_order_patterns() {
    let r = date(2024, 1, 15);

    // Empty & ConstPoints (case-insensitive)
    assert_eq!(point_sort_order("", r), Some(0.0));
    assert_eq!(point_sort_order("O/N", r), Some(0.0));
    assert_eq!(point_sort_order("o/n", r), Some(0.0));
    assert_eq!(point_sort_order("T/N", r), Some(0.1));
    assert_eq!(point_sort_order("OIS FIX", r), Some(1.0));
    assert_eq!(point_sort_order("Cash Stub", r), Some(1.1));
    assert_eq!(point_sort_order("CashStub", r), Some(1.1));
    assert_eq!(point_sort_order("DEFAULT", r), Some(0.0));
    assert_eq!(point_sort_order("In", r), Some(0.1));
    assert_eq!(point_sort_order("Out", r), Some(0.2));

    // Compound semicolon points
    assert_eq!(point_sort_order("1y;5y", r), Some(365.5));
    assert_eq!(point_sort_order(" 1y ; 2y ; 3y ", r), Some(365.5));
    assert!((point_sort_order("6m;2y;10y", r).unwrap() - 182.433_333_333_333_34).abs() < 1e-12);
    assert_eq!(point_sort_order("O/N;5y", r), Some(0.0));
    assert_eq!(point_sort_order("USD Rho;5y", r), Some(0.0));

    // infl_vol, Copula (None), SeasonalFront
    assert_eq!(point_sort_order("USD Caplet 1y", r), Some(0.0));
    assert_eq!(point_sort_order("ZCCap 5y", r), Some(1.0));
    assert_eq!(point_sort_order("Swaption 10y", r), Some(2.0));
    assert_eq!(point_sort_order("ZCSwo Caplet", r), Some(3.0));
    for cop in [
        "USD Rho",
        "EUR KO",
        "Rho Rate 1y",
        "Rho Vol 5y",
        "1y RR",
        "BF 2y",
        "Alpha",
        "Beta",
        "K0=L",
        "K0=S",
    ] {
        assert_eq!(point_sort_order(cop, r), None, "copula {cop}");
    }
    assert_eq!(point_sort_order("Front", r), Some(0.0));
    assert_eq!(point_sort_order("Back", r), Some(1.0));
    assert_eq!(point_sort_order("Back Front", r), Some(1.0));
    assert_eq!(point_sort_order("Front Back", r), Some(0.0));

    // MMM & EuroOrFra with POSIX %y 68/69 pivot
    assert_eq!(point_sort_order("Jan", r), Some(-8780.0));
    assert_eq!(point_sort_order("Dec", r), Some(-8445.0));
    assert_eq!(point_sort_order("JAN25", r), Some(366.0));
    assert_eq!(point_sort_order("Dec68", r), Some(16406.0)); // 2068-12-15
    assert_eq!(point_sort_order("Dec69", r), Some(-19754.0)); // 1969-12-15

    // RDatePart, CashFX, PricerBF, FRAx
    assert_eq!(point_sort_order("1d", r), Some(1.0));
    assert_eq!(point_sort_order("10B", r), Some(10.0));
    assert_eq!(point_sort_order("-2w", r), Some(-14.0));
    assert_eq!(point_sort_order("2F", r), Some(60.0));
    assert_eq!(point_sort_order("1y5y", r), Some(365.0));
    assert_eq!(point_sort_order("1y5y XC", r), Some(365.0));
    assert_eq!(point_sort_order("1y5y10y", r), Some(365.0));
    assert_eq!(point_sort_order("3x6", r), Some(90.0));

    // SpikeQE, MMMYYYY, DDMMMYYYY (including leap 29Feb2024 and invalid calendar fallback)
    assert_eq!(point_sort_order("QE1-2025", r), Some(411.0));
    assert_eq!(point_sort_order("QE2-2025", r), Some(503.0));
    assert_eq!(point_sort_order("QE3-2025", r), Some(595.0));
    assert_eq!(point_sort_order("QE4-2025", r), Some(686.0));
    assert_eq!(point_sort_order("Jun2025", r), Some(503.0));
    assert_eq!(point_sort_order("15Mar2026", r), Some(790.0));
    assert_eq!(point_sort_order("29Feb2024", r), Some(45.0));
    assert_eq!(point_sort_order("1Jan2024", r), Some(-14.0));
    assert_eq!(point_sort_order("31Feb2024", r), Some(0.0)); // invalid date -> safe fallback

    // Number, FloatingYear, PricerCoordI/II, PricerBondSpread
    assert_eq!(point_sort_order("42", r), Some(42.0));
    assert_eq!(point_sort_order("0.5y", r), Some(182.5));
    assert_eq!(point_sort_order("1.5Y", r), Some(547.5));
    assert_eq!(point_sort_order(".5y", r), Some(182.5));
    assert_eq!(point_sort_order("No 5", r), Some(5.0));
    assert_eq!(point_sort_order("Pricer 12", r), Some(12.0));
    assert_eq!(point_sort_order("5S25/5Q25", r), None);

    // LYY, DatePair, DatePair2, MMMYY, FXVolAddonParms
    assert_eq!(point_sort_order("H25", r), Some(411.0));
    assert_eq!(point_sort_order("M26", r), Some(868.0));
    assert_eq!(point_sort_order("Z68", r), Some(16392.0));
    assert_eq!(point_sort_order("Z69", r), Some(-19768.0));
    assert_eq!(point_sort_order("20240101/20250615", r), Some(517.0));
    assert_eq!(point_sort_order("20240101 20250615", r), Some(517.0));
    assert_eq!(point_sort_order("JAN 25", r), Some(352.0));
    assert_eq!(point_sort_order("Spread Addon", r), None);
    assert_eq!(point_sort_order("HL", r), None);

    // BondCoord, BondFut, FFFut, RepoGC, Relative, DDMMMYY, fallback
    assert_eq!(point_sort_order("1 2.5 15/03/2026", r), Some(790.0));
    assert_eq!(point_sort_order("USTH4", r), Some(46.0));
    assert_eq!(point_sort_order("FFZ5", r), Some(321.0));
    assert_eq!(point_sort_order("FFH4", r), Some(46.0));
    assert_eq!(point_sort_order("ON GC", r), Some(0.0));
    assert_eq!(point_sort_order("TN GC", r), Some(1.0));
    assert_eq!(point_sort_order("SN GC", r), Some(2.0));
    assert_eq!(point_sort_order("3 Month GC", r), Some(90.0));
    assert_eq!(point_sort_order("2 Week GC", r), Some(14.0));
    assert_eq!(point_sort_order("1 Year GC", r), Some(365.0));
    assert_eq!(point_sort_order("5 Day GC", r), Some(5.0));
    assert_eq!(point_sort_order("10 day", r), Some(10.0));
    assert_eq!(point_sort_order("3 week", r), Some(21.0));
    assert_eq!(point_sort_order("6 month", r), Some(180.0));
    assert_eq!(point_sort_order("2 YEAR", r), Some(730.0));
    assert_eq!(point_sort_order("05Feb25", r), Some(387.0));
    assert_eq!(point_sort_order("29Feb24", r), Some(45.0));
    assert_eq!(point_sort_order("unknown_point", r), Some(0.0));
}

#[test]
fn risk_sort_risk_multi_column_stability_and_none() {
    let r = date(2024, 1, 15);

    // Multi-column ordering: date -> time -> mkt_type -> mkt_asset -> mkt_class -> mkt_point
    let table = RiskTable {
        unit: Some("USD".into()),
        rows: vec![
            risk_row(
                Some(date(2024, 1, 2)),
                Some("10:00"),
                "IR",
                "USD",
                "SWAP",
                "1y",
                1.0,
            ),
            risk_row(
                Some(date(2024, 1, 1)),
                Some("14:00"),
                "IR",
                "USD",
                "SWAP",
                "5y",
                2.0,
            ),
            risk_row(
                Some(date(2024, 1, 1)),
                Some("09:00"),
                "IR",
                "USD",
                "SWAP",
                "10y",
                3.0,
            ),
            risk_row(
                Some(date(2024, 1, 1)),
                Some("09:00"),
                "FX",
                "USD",
                "FWD",
                "5y",
                4.0,
            ),
            risk_row(
                Some(date(2024, 1, 1)),
                Some("09:00"),
                "IR",
                "EUR",
                "SWAP",
                "5y",
                5.0,
            ),
            risk_row(
                Some(date(2024, 1, 1)),
                Some("09:00"),
                "IR",
                "USD",
                "CASH",
                "5y",
                6.0,
            ),
            risk_row(
                Some(date(2024, 1, 1)),
                Some("09:00"),
                "IR",
                "USD",
                "SWAP",
                "3m",
                7.0,
            ),
        ],
    };
    let sorted = sort_risk(&table, r);
    assert_eq!(sorted.unit.as_deref(), Some("USD"));
    let values: Vec<f64> = sorted.rows.iter().map(|x| x.value).collect();
    assert_eq!(values, vec![4.0, 5.0, 6.0, 7.0, 3.0, 2.0, 1.0]);

    // Stability on tied sort orders ("30d" and "1m" both 30.0) and None -> 0.0 handling
    let tied = RiskTable {
        unit: None,
        rows: vec![
            risk_row(None, None, "IR", "USD", "SWAP", "1y", 10.0),
            risk_row(None, None, "IR", "USD", "SWAP", "30d", 20.0),
            risk_row(None, None, "IR", "USD", "SWAP", "1m", 30.0),
            risk_row(None, None, "IR", "USD", "SWAP", "USD Rho", 40.0),
            risk_row(None, None, "IR", "USD", "SWAP", "O/N", 50.0),
            risk_row(None, None, "IR", "USD", "SWAP", "-1w", 60.0),
        ],
    };
    let sorted_tied = sort_risk(&tied, r);
    let pts: Vec<&str> = sorted_tied
        .rows
        .iter()
        .map(|x| x.mkt_point.as_str())
        .collect();
    assert_eq!(pts, vec!["-1w", "USD Rho", "O/N", "30d", "1m", "1y"]);
}

#[test]
fn risk_aggregate_risk_grouping_thresholds_units_and_errors() {
    let r = date(2024, 1, 15);

    // Multi-table grouping, duplicates within single table, fillna(0) on NaN, ±Inf propagation
    let t1 = RiskTable {
        unit: None,
        rows: vec![
            risk_row(None, None, "IR", "USD", "SWAP", "5y", 10.0),
            risk_row(None, None, "IR", "USD", "SWAP", "5y", 15.0),
            risk_row(None, None, "IR", "USD", "SWAP", "30d", 5.0),
            risk_row(None, None, "IR", "USD", "SWAP", "1m", 7.0),
            risk_row(None, None, "IR", "USD", "SWAP", "1y", f64::NAN),
            risk_row(None, None, "IR", "USD", "SWAP", "2y", -10.0),
            risk_row(None, None, "IR", "USD", "SWAP", "10y", f64::INFINITY),
        ],
    };
    let t2 = RiskTable {
        unit: Some("USD".into()),
        rows: vec![
            risk_row(None, None, "IR", "USD", "SWAP", "5y", -25.0), // exact cancellation to 0.0
            risk_row(None, None, "IR", "USD", "SWAP", "1y", 12.5),
            risk_row(None, None, "IR", "USD", "SWAP", "2y", -0.5), // -10.5
        ],
    };

    let agg_all = aggregate_risk(&[t1.clone(), t2.clone()], None, r).unwrap();
    // Unit inherited from t2 despite t1 having None
    assert_eq!(agg_all.unit.as_deref(), Some("USD"));
    // groupby pre-orders lexicographically before sort_risk -> "1m" precedes "30d"!
    let pts: Vec<&str> = agg_all.rows.iter().map(|x| x.mkt_point.as_str()).collect();
    assert_eq!(pts, vec!["1m", "30d", "1y", "2y", "5y", "10y"]);
    assert_eq!(agg_all.rows[0].value, 7.0);
    assert_eq!(agg_all.rows[1].value, 5.0);
    assert_eq!(agg_all.rows[2].value, 12.5);
    assert_eq!(agg_all.rows[3].value, -10.5);
    assert_eq!(agg_all.rows[4].value, 0.0);
    assert!(agg_all.rows[5].value.is_infinite() && agg_all.rows[5].value.is_sign_positive());

    // Strict abs(value) > threshold boundaries
    let th0 = aggregate_risk(&[t1.clone(), t2.clone()], Some(0.0), r).unwrap();
    assert_eq!(th0.rows.len(), 5); // 5y (0.0) dropped
    let th105 = aggregate_risk(&[t1.clone(), t2.clone()], Some(10.5), r).unwrap();
    // |-10.5| == 10.5 is NOT strictly greater -> dropped! Only 1y (12.5) and 10y (+Inf) survive.
    let th105_pts: Vec<&str> = th105.rows.iter().map(|x| x.mkt_point.as_str()).collect();
    assert_eq!(th105_pts, vec!["1y", "10y"]);
    let th_neg = aggregate_risk(&[t1.clone(), t2.clone()], Some(-1.0), r).unwrap();
    assert_eq!(th_neg.rows.len(), 6); // 0.0 kept when threshold is negative
    let th_nan = aggregate_risk(&[t1.clone(), t2.clone()], Some(f64::NAN), r).unwrap();
    assert!(th_nan.rows.is_empty());
    let th_inf = aggregate_risk(&[t1.clone(), t2.clone()], Some(f64::INFINITY), r).unwrap();
    assert!(th_inf.rows.is_empty());

    // Empty unit string treated as unset; mismatched non-empty units return Error::UnitMismatch
    let t_empty_unit = RiskTable {
        unit: Some(String::new()),
        rows: vec![risk_row(None, None, "IR", "USD", "SWAP", "1y", 1.0)],
    };
    let ok_inherit = aggregate_risk(&[t_empty_unit, t2.clone()], None, r).unwrap();
    assert_eq!(ok_inherit.unit.as_deref(), Some("USD"));

    let t_eur = RiskTable {
        unit: Some("EUR".into()),
        rows: vec![risk_row(None, None, "IR", "EUR", "SWAP", "1y", 1.0)],
    };
    assert_eq!(
        aggregate_risk(&[t2, t_eur], None, r),
        Err(Error::UnitMismatch)
    );

    // Empty slice -> Error::Length
    assert_eq!(aggregate_risk(&[], None, r), Err(Error::Length));
}

#[test]
fn risk_subtract_risk_cancellation_schema_and_determinism() {
    let r = date(2024, 1, 15);

    let left = RiskTable {
        unit: Some("USD".into()),
        rows: vec![
            risk_row(None, None, "IR", "USD", "SWAP", "5y", 10.0),
            risk_row(None, None, "IR", "USD", "SWAP", "1y", 25.0),
            risk_row(None, None, "IR", "USD", "SWAP", "3m", 4.0),
        ],
    };
    let right = RiskTable {
        unit: Some("USD".into()),
        rows: vec![
            risk_row(None, None, "IR", "USD", "SWAP", "5y", 10.0), // exact cancel -> 0.0
            risk_row(None, None, "IR", "USD", "SWAP", "2y", 6.0),  // asymmetric -> -6.0
            risk_row(None, None, "IR", "USD", "SWAP", "1y", f64::NAN), // NaN -> 0.0 -> 25.0
        ],
    };

    let diff1 = subtract_risk(&left, &right, r).unwrap();
    let diff2 = subtract_risk(&left, &right, r).unwrap();
    assert_eq!(diff1, diff2);
    assert_eq!(diff1.unit.as_deref(), Some("USD"));
    let summary: Vec<(&str, f64)> = diff1
        .rows
        .iter()
        .map(|x| (x.mkt_point.as_str(), x.value))
        .collect();
    assert_eq!(
        summary,
        vec![("3m", 4.0), ("1y", 25.0), ("2y", -6.0), ("5y", 0.0)]
    );

    // Unit mismatch -> Error::UnitMismatch
    let right_eur = RiskTable {
        unit: Some("EUR".into()),
        rows: right.rows.clone(),
    };
    assert_eq!(
        subtract_risk(&left, &right_eur, r),
        Err(Error::UnitMismatch)
    );

    // Date presence mismatch -> Error::Length
    let dated = RiskTable {
        unit: Some("USD".into()),
        rows: vec![risk_row(
            Some(date(2024, 1, 15)),
            None,
            "IR",
            "USD",
            "SWAP",
            "5y",
            1.0,
        )],
    };
    assert_eq!(subtract_risk(&left, &dated, r), Err(Error::Length));

    // Time presence mismatch -> Error::Length
    let timed = RiskTable {
        unit: Some("USD".into()),
        rows: vec![risk_row(
            None,
            Some("09:30"),
            "IR",
            "USD",
            "SWAP",
            "5y",
            1.0,
        )],
    };
    assert_eq!(subtract_risk(&left, &timed, r), Err(Error::Length));

    // Empty right table succeeds and preserves left rows sorted
    let empty_right = RiskTable {
        unit: None,
        rows: vec![],
    };
    let diff_empty = subtract_risk(&left, &empty_right, r).unwrap();
    assert_eq!(diff_empty.rows.len(), 3);
    assert_eq!(diff_empty.rows[0].mkt_point, "3m");
}

#[test]
fn vol_swap_volatility_equivalence_defaults_factors_zero_mean_and_leap_day() {
    // 50-day synthetic price series (reproducing upstream test_vol_swap_volatility coverage)
    let mut dates = Vec::with_capacity(50);
    let mut vals = Vec::with_capacity(50);
    let mut price = 100.0_f64;
    for i in 0..50usize {
        let month = if i < 31 { 1 } else { 2 };
        let day = if i < 31 { i + 1 } else { i - 30 };
        dates.push(date(2025, month, day as u8));
        if i > 0 {
            let r = 0.008 * ((i as f64) * 1.37).sin() + 0.004 * ((i as f64) * 0.53).cos();
            price *= r.exp();
        }
        vals.push(price);
    }
    let px = Series::try_new(dates, vals).unwrap();

    // Exact equivalence with volatility(..., Window(n, n - 1), Log, Some(af), zm)
    for n_days in [3usize, 22, 25, 49, 50] {
        let w = Window::Observations {
            size: n_days,
            ramp: n_days - 1,
        };
        for (af, zm) in [
            (None, true),
            (Some(252), true),
            (Some(365), true),
            (None, false),
            (Some(0), true),
        ] {
            let wrap = vol_swap_volatility(&px, Some(n_days), af, zm).unwrap();
            let direct = volatility(
                &px,
                w,
                Some(ReturnKind::Logarithmic),
                Some(af.unwrap_or(252)),
                zm,
            )
            .unwrap();
            assert_eq!(wrap.dates(), direct.dates());
            for (&a, &b) in wrap.values().iter().zip(direct.values()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    // Default n_days = None matches Some(px.len())
    let def_zm = vol_swap_volatility(&px, None, None, true).unwrap();
    let full_zm = vol_swap_volatility(&px, Some(50), Some(252), true).unwrap();
    assert_eq!(def_zm, full_zm);
    assert_eq!(def_zm.dates().len(), 1);
    assert_eq!(def_zm.dates()[0], date(2025, 2, 19));

    // Custom annualization_factor 365 scales by sqrt(365/252); factor 0 scales finite values to 0.0
    let v252 = vol_swap_volatility(&px, Some(22), None, true).unwrap();
    let v365 = vol_swap_volatility(&px, Some(22), Some(365), true).unwrap();
    let v0 = vol_swap_volatility(&px, Some(22), Some(0), true).unwrap();
    let expected_ratio = (365.0_f64 / 252.0_f64).sqrt();
    for (&a, &b) in v252.values().iter().zip(v365.values()) {
        assert!(((b / a) - expected_ratio).abs() < 1e-12);
    }
    assert!(v0.values().iter().all(|&v| v == 0.0));

    // assume_zero_mean = true vs false differ and match explicit RMS / sample formulas
    let v_sample = vol_swap_volatility(&px, Some(22), None, false).unwrap();
    assert!((v252.values()[0] - v_sample.values()[0]).abs() > 1e-6);

    let log_rets: Vec<f64> = px
        .values()
        .windows(2)
        .map(|w| w[1].ln() - w[0].ln())
        .collect();
    let win_rets = &log_rets[..21]; // 22 price days -> 21 log returns
    let rms =
        (win_rets.iter().map(|r| r * r).sum::<f64>() / 21.0).sqrt() * 252.0_f64.sqrt() * 100.0;
    let mean = win_rets.iter().sum::<f64>() / 21.0;
    let sample = (win_rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 20.0).sqrt()
        * 252.0_f64.sqrt()
        * 100.0;
    assert!((v252.values()[0] - rms).abs() < 1e-12);
    assert!((v_sample.values()[0] - sample).abs() < 1e-12);

    // Leap day 2020-02-29 preserved
    let leap = Series::try_new(
        vec![
            date(2020, 2, 27),
            date(2020, 2, 28),
            date(2020, 2, 29),
            date(2020, 3, 1),
            date(2020, 3, 2),
        ],
        vec![100.0, 102.0, 101.0, 104.0, 103.5],
    )
    .unwrap();
    let leap_vol = vol_swap_volatility(&leap, Some(3), None, true).unwrap();
    assert_eq!(
        leap_vol.dates(),
        &[date(2020, 2, 29), date(2020, 3, 1), date(2020, 3, 2)]
    );
    assert!(leap_vol.values().iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn vol_swap_volatility_boundaries_empty_singleton_oversize_nan_and_determinism() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    // Empty input returns empty BEFORE validating n_days (including n_days = Some(0) and Some(5))
    for n_days in [None, Some(0), Some(5)] {
        for zm in [true, false] {
            let res = vol_swap_volatility(&empty, n_days, None, zm).unwrap();
            assert!(res.dates().is_empty() && res.values().is_empty());
        }
    }

    // Singleton -> [NaN] for both zero-mean and sample
    let single = Series::try_new(vec![date(2024, 1, 1)], vec![100.0]).unwrap();
    for zm in [true, false] {
        let res = vol_swap_volatility(&single, None, None, zm).unwrap();
        assert_eq!(res.dates(), &[date(2024, 1, 1)]);
        assert!(res.values()[0].is_nan());
    }

    let dates: Vec<_> = (1..=6).map(|d| date(2024, 1, d)).collect();
    let px = Series::try_new(
        dates.clone(),
        vec![100.0, 102.0, 101.0, 104.0, 103.0, 106.0],
    )
    .unwrap();

    // n_days = Some(0) -> Err(Error::Window)
    assert_eq!(
        vol_swap_volatility(&px, Some(0), None, true),
        Err(Error::Window)
    );
    assert_eq!(
        vol_swap_volatility(&px, Some(0), None, false),
        Err(Error::Window)
    );

    // n_days = Some(1) -> Window(1, 0): first NaN, remaining sqrt(af) * |r_i| * 100 for zm=true, all NaN for zm=false
    let n1_zm = vol_swap_volatility(&px, Some(1), Some(252), true).unwrap();
    assert_eq!(n1_zm.dates(), px.dates());
    assert!(n1_zm.values()[0].is_nan());
    for i in 1..6 {
        let r = (px.values()[i].ln() - px.values()[i - 1].ln()).abs();
        let expected = 252.0_f64.sqrt() * r * 100.0;
        assert!((n1_zm.values()[i] - expected).abs() < 1e-12);
    }
    let n1_sample = vol_swap_volatility(&px, Some(1), Some(252), false).unwrap();
    assert_eq!(n1_sample.dates(), px.dates());
    assert!(n1_sample.values().iter().all(|v| v.is_nan()));

    // n_days = Some(2) -> Window(2, 1): index 0 of output is finite for zm=true, NaN for zm=false
    let n2_zm = vol_swap_volatility(&px, Some(2), None, true).unwrap();
    let n2_sample = vol_swap_volatility(&px, Some(2), None, false).unwrap();
    assert_eq!(n2_zm.dates(), &dates[1..]);
    assert!(n2_zm.values().iter().all(|v| v.is_finite()));
    assert!(n2_sample.values()[0].is_nan());
    assert!(n2_sample.values()[1..].iter().all(|v| v.is_finite()));

    // Oversize: len + 1 returns empty, len + 2 returns Err(Error::Ramp)
    let ov1 = vol_swap_volatility(&px, Some(7), None, true).unwrap();
    assert!(ov1.dates().is_empty() && ov1.values().is_empty());
    assert_eq!(
        vol_swap_volatility(&px, Some(8), None, true),
        Err(Error::Ramp)
    );

    // Constant prices -> 0.0
    let const_px = Series::try_new(dates.clone(), vec![100.0; 6]).unwrap();
    for zm in [true, false] {
        let res = vol_swap_volatility(&const_px, Some(3), None, zm).unwrap();
        assert_eq!(res.dates(), &dates[2..]);
        assert!(res.values().iter().all(|&v| v == 0.0));
    }

    // NaN propagation: mid-series NaN creates two NaN log returns
    let nan_px = Series::try_new(
        dates.clone(),
        vec![100.0, 102.0, f64::NAN, 104.0, 103.0, 106.0],
    )
    .unwrap();
    let nan_zm = vol_swap_volatility(&nan_px, Some(3), None, true).unwrap();
    let nan_sample = vol_swap_volatility(&nan_px, Some(3), None, false).unwrap();
    assert_eq!(nan_zm.dates(), &dates[2..]);
    assert!(nan_zm.values()[0].is_finite()); // window [NaN, r1, NaN] has 1 return
    assert!(nan_zm.values()[1].is_finite()); // window [r1, NaN, NaN] has 1 return
    assert!(nan_sample.values()[0].is_nan()); // count == 1 < 2
    assert!(nan_sample.values()[1].is_nan()); // count == 1 < 2
    assert!(nan_sample.values()[2].is_nan()); // window [NaN, NaN, r4] has 1 return
    assert!(nan_sample.values()[3].is_finite()); // window [NaN, r4, r5] has 2 returns

    // Non-positive prices -> Err(Error::NonPositivePrice)
    let zero_px = Series::try_new(dates[..3].to_vec(), vec![100.0, 0.0, 102.0]).unwrap();
    let neg_px = Series::try_new(dates[..3].to_vec(), vec![100.0, -1.0, 102.0]).unwrap();
    assert_eq!(
        vol_swap_volatility(&zero_px, Some(2), None, true),
        Err(Error::NonPositivePrice)
    );
    assert_eq!(
        vol_swap_volatility(&neg_px, Some(2), None, false),
        Err(Error::NonPositivePrice)
    );

    // Bit-for-bit determinism across repeated calls
    let run1 = vol_swap_volatility(&px, Some(4), Some(252), true).unwrap();
    let run2 = vol_swap_volatility(&px, Some(4), Some(252), true).unwrap();
    assert_eq!(run1.dates(), run2.dates());
    for (&a, &b) in run1.values().iter().zip(run2.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn corr_swap_correlation_equivalence_defaults_misaligned_and_leap_day() {
    let dates: Vec<_> = (1..=10).map(|d| date(2025, 1, d)).collect();
    let x = Series::try_new(
        dates.clone(),
        vec![
            100.0, 101.8, 100.5, 103.2, 102.9, 105.4, 104.1, 106.7, 108.0, 107.2,
        ],
    )
    .unwrap();
    let y = Series::try_new(
        dates.clone(),
        vec![50.0, 51.2, 50.9, 52.4, 51.8, 53.0, 53.6, 54.1, 55.3, 54.4],
    )
    .unwrap();

    // Upstream test_corr_swap_correlation equivalence for assume_zero_mean = false across window sizes
    for w in [3usize, 5, 9] {
        let direct = correlation(
            &x,
            &y,
            Window::Observations {
                size: w,
                ramp: w - 1,
            },
            Some((ReturnKind::Logarithmic, ReturnKind::Logarithmic)),
            false,
        )
        .unwrap();
        let wrap = corr_swap_correlation(&x, &y, Some(w + 1), false).unwrap();
        assert_eq!(wrap.dates(), direct.dates());
        for (&a, &b) in wrap.values().iter().zip(direct.values()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        // assume_zero_mean = true yields distinct values in [-1, 1] matching manual formula
        let zm = corr_swap_correlation(&x, &y, Some(w + 1), true).unwrap();
        assert_eq!(zm.dates(), wrap.dates());
        assert!(zm.values()[0].is_nan()); // masked by inner ramp on aligned returns
        for &v in &zm.values()[1..] {
            assert!(v.is_finite() && (-1.0..=1.0).contains(&v));
        }
        assert!((zm.values()[1] - wrap.values()[1]).abs() > 1e-4);

        let rx: Vec<f64> = x
            .values()
            .windows(2)
            .map(|p| p[1].ln() - p[0].ln())
            .collect();
        let ry: Vec<f64> = y
            .values()
            .windows(2)
            .map(|p| p[1].ln() - p[0].ln())
            .collect();
        let cov: f64 = rx[..w]
            .iter()
            .zip(&ry[..w])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (w as f64);
        let vx: f64 = rx[..w].iter().map(|a| a * a).sum::<f64>() / (w as f64);
        let vy: f64 = ry[..w].iter().map(|b| b * b).sum::<f64>() / (w as f64);
        let manual = cov / (vx.sqrt() * vy.sqrt());
        assert!((zm.values()[1] - manual).abs() < 1e-12);
    }

    // Default n_days = None matches Some(x.len().min(y.len()))
    let def_eq = corr_swap_correlation(&x, &y, None, true).unwrap();
    let exp_eq = corr_swap_correlation(&x, &y, Some(10), true).unwrap();
    assert_eq!(def_eq.dates(), exp_eq.dates());
    for (&a, &b) in def_eq.values().iter().zip(exp_eq.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // Unequal lengths: default n_days uses min(x.len(), y.len())
    let y_short = Series::try_new(dates[..6].to_vec(), y.values()[..6].to_vec()).unwrap();
    let def_short = corr_swap_correlation(&x, &y_short, None, false).unwrap();
    let exp_short = corr_swap_correlation(&x, &y_short, Some(6), false).unwrap();
    assert_eq!(def_short.dates(), exp_short.dates());
    for (&a, &b) in def_short.values().iter().zip(exp_short.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // Misaligned calendars: unmatched date in x becomes NaN
    let mx = Series::try_new(
        vec![
            date(2024, 3, 1),
            date(2024, 3, 2),
            date(2024, 3, 3),
            date(2024, 3, 4),
            date(2024, 3, 5),
            date(2024, 3, 6),
        ],
        vec![100.0, 102.0, 104.0, 103.0, 106.0, 108.0],
    )
    .unwrap();
    let my = Series::try_new(
        vec![
            date(2024, 3, 1),
            date(2024, 3, 3),
            date(2024, 3, 4),
            date(2024, 3, 5),
            date(2024, 3, 6),
        ],
        vec![200.0, 206.0, 204.0, 210.0, 215.0],
    )
    .unwrap();
    let mis_f = corr_swap_correlation(&mx, &my, Some(4), false).unwrap();
    assert_eq!(mis_f.dates(), &mx.dates()[2..]);
    assert!(mis_f.values()[0].is_nan()); // only 1 aligned non-NaN return pair so far
    assert!(mis_f.values()[1..].iter().all(|v| v.is_finite()));

    // Leap day 2020-02-29 preserved
    let leap_dates = vec![
        date(2020, 2, 27),
        date(2020, 2, 28),
        date(2020, 2, 29),
        date(2020, 3, 1),
        date(2020, 3, 2),
        date(2020, 3, 3),
    ];
    let lx = Series::try_new(
        leap_dates.clone(),
        vec![100.0, 102.0, 101.0, 104.0, 103.0, 106.0],
    )
    .unwrap();
    let ly = Series::try_new(leap_dates.clone(), vec![50.0, 51.5, 50.2, 52.0, 52.5, 54.0]).unwrap();
    let leap_zm = corr_swap_correlation(&lx, &ly, Some(4), true).unwrap();
    assert_eq!(leap_zm.dates(), &leap_dates[2..]);
    assert_eq!(leap_zm.dates()[0], date(2020, 2, 29));
    assert!(leap_zm.values()[0].is_nan());
    assert!(leap_zm.values()[1..]
        .iter()
        .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
}

#[test]
fn corr_swap_correlation_boundaries_empty_singleton_irregular_constant_and_determinism() {
    let empty = Series::try_new(vec![], vec![]).unwrap();
    let dates: Vec<_> = (1..=6).map(|d| date(2024, 1, d)).collect();
    let x = Series::try_new(
        dates.clone(),
        vec![100.0, 102.0, 101.0, 104.0, 103.0, 106.0],
    )
    .unwrap();
    let y = Series::try_new(dates.clone(), vec![50.0, 51.5, 50.2, 52.0, 52.5, 54.0]).unwrap();

    // Empty x combinations return empty BEFORE validating n_days
    for n_days in [None, Some(0), Some(1), Some(5)] {
        for zm in [true, false] {
            let ee = corr_swap_correlation(&empty, &empty, n_days, zm).unwrap();
            assert!(ee.dates().is_empty() && ee.values().is_empty());
            let ey = corr_swap_correlation(&empty, &y, n_days, zm).unwrap();
            assert!(ey.dates().is_empty() && ey.values().is_empty());
        }
    }

    // Non-empty x, empty y
    assert_eq!(
        corr_swap_correlation(&x, &empty, None, true),
        Err(Error::Window)
    );
    assert_eq!(
        corr_swap_correlation(&x, &empty, Some(4), true),
        Err(Error::AnnualizationFactor)
    );
    let x_empty_f = corr_swap_correlation(&x, &empty, Some(4), false).unwrap();
    assert_eq!(x_empty_f.dates(), &dates[2..]);
    assert!(x_empty_f.values().iter().all(|v| v.is_nan()));

    // Singleton -> Err(Error::Window)
    let single = Series::try_new(vec![date(2024, 1, 1)], vec![100.0]).unwrap();
    assert_eq!(
        corr_swap_correlation(&single, &single, None, true),
        Err(Error::Window)
    );

    // n_days = Some(0) and Some(1) -> Err(Error::Window)
    for bad in [0usize, 1] {
        assert_eq!(
            corr_swap_correlation(&x, &y, Some(bad), true),
            Err(Error::Window)
        );
        assert_eq!(
            corr_swap_correlation(&x, &y, Some(bad), false),
            Err(Error::Window)
        );
    }

    // n_days = Some(2) -> Window(1, 0): zm=true yields NaN then +-1.0; zm=false yields all NaN
    let n2_zm = corr_swap_correlation(&x, &y, Some(2), true).unwrap();
    assert_eq!(n2_zm.dates(), x.dates());
    assert!(n2_zm.values()[0].is_nan());
    for &v in &n2_zm.values()[1..] {
        assert!((v.abs() - 1.0).abs() < 1e-12);
    }
    let n2_f = corr_swap_correlation(&x, &y, Some(2), false).unwrap();
    assert!(n2_f.values().iter().all(|v| v.is_nan()));

    // 2-point series with n_days = None: zm=true -> Error::AnnualizationFactor, zm=false -> [NaN, NaN]
    let two_x = Series::try_new(dates[..2].to_vec(), vec![100.0, 102.0]).unwrap();
    let two_y = Series::try_new(dates[..2].to_vec(), vec![50.0, 51.0]).unwrap();
    assert_eq!(
        corr_swap_correlation(&two_x, &two_y, None, true),
        Err(Error::AnnualizationFactor)
    );
    let two_f = corr_swap_correlation(&two_x, &two_y, None, false).unwrap();
    assert_eq!(two_f.dates(), &dates[..2]);
    assert!(two_f.values().iter().all(|v| v.is_nan()));

    // Irregular 3-day spacing: zm=true -> Error::AnnualizationFactor; zm=false -> finite Pearson
    let irreg_dates = vec![
        date(2024, 1, 1),
        date(2024, 1, 4),
        date(2024, 1, 7),
        date(2024, 1, 10),
        date(2024, 1, 13),
    ];
    let ix = Series::try_new(irreg_dates.clone(), vec![100.0, 102.0, 101.0, 104.0, 103.0]).unwrap();
    let iy = Series::try_new(irreg_dates.clone(), vec![50.0, 51.0, 50.5, 52.0, 51.5]).unwrap();
    assert_eq!(
        corr_swap_correlation(&ix, &iy, Some(4), true),
        Err(Error::AnnualizationFactor)
    );
    let irreg_f = corr_swap_correlation(&ix, &iy, Some(4), false).unwrap();
    assert_eq!(irreg_f.dates(), &irreg_dates[2..]);
    assert!(irreg_f.values().iter().all(|&v| (v - 1.0).abs() < 1e-12));

    // Constant x (both constant price 100.0 and constant log-return 100.0 * 1.25^i) -> NaN for zm=false
    let const_px = Series::try_new(dates.clone(), vec![100.0; 6]).unwrap();
    let const_lr = Series::try_new(
        dates.clone(),
        vec![100.0, 125.0, 156.25, 195.3125, 244.140625, 305.17578125],
    )
    .unwrap();
    for cx in [&const_px, &const_lr] {
        let cf = corr_swap_correlation(cx, &y, Some(4), false).unwrap();
        assert_eq!(cf.dates(), &dates[2..]);
        assert!(cf.values().iter().all(|v| v.is_nan()));
    }
    let const_px_zm = corr_swap_correlation(&const_px, &y, Some(4), true).unwrap();
    assert!(const_px_zm.values().iter().all(|v| v.is_nan()));
    let const_lr_zm = corr_swap_correlation(&const_lr, &y, Some(4), true).unwrap();
    assert!(const_lr_zm.values()[0].is_nan());
    assert!(const_lr_zm.values()[1..].iter().all(|v| v.is_finite()));

    // Oversize n_days boundaries (x.len() == 6)
    assert_eq!(
        corr_swap_correlation(&x, &y, Some(7), true),
        Err(Error::AnnualizationFactor)
    );
    let ov7_f = corr_swap_correlation(&x, &y, Some(7), false).unwrap();
    assert_eq!(ov7_f.dates(), &[date(2024, 1, 6)]);
    assert!(ov7_f.values()[0].is_finite());
    assert_eq!(
        corr_swap_correlation(&x, &y, Some(8), true),
        Err(Error::Ramp)
    );
    let ov8_f = corr_swap_correlation(&x, &y, Some(8), false).unwrap();
    assert!(ov8_f.dates().is_empty());
    assert_eq!(
        corr_swap_correlation(&x, &y, Some(9), false),
        Err(Error::Ramp)
    );

    // Non-positive price in x or y -> Err(Error::NonPositivePrice)
    let nonpos = Series::try_new(dates[..4].to_vec(), vec![100.0, 0.0, 102.0, 103.0]).unwrap();
    let ok4 = Series::try_new(dates[..4].to_vec(), vec![50.0, 51.0, 52.0, 53.0]).unwrap();
    assert_eq!(
        corr_swap_correlation(&nonpos, &ok4, Some(3), true),
        Err(Error::NonPositivePrice)
    );
    assert_eq!(
        corr_swap_correlation(&ok4, &nonpos, Some(3), false),
        Err(Error::NonPositivePrice)
    );

    // Bit-for-bit determinism across repeated calls
    let run1 = corr_swap_correlation(&x, &y, Some(4), true).unwrap();
    let run2 = corr_swap_correlation(&x, &y, Some(4), true).unwrap();
    assert_eq!(run1.dates(), run2.dates());
    for (&a, &b) in run1.values().iter().zip(run2.values()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
