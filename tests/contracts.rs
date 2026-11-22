use quantus_math::{
    beta, correlation, exponential_moving_average, max_drawdown, moving_average, returns,
    volatility, Date, Error, ReturnKind, Series, Window,
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
