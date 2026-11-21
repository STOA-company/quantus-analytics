use quantus_math::{moving_average, returns, Date, Error, ReturnKind, Series, Window};
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
