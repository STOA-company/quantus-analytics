use quantus_math::{
    correlation, moving_average, returns, volatility, Date, Error, ReturnKind, Series, Window,
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
