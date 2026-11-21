use quantus_math::{moving_average, returns, Date, ReturnKind, Series, Window};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dates = (1..=6)
        .map(|day| Date::new(2019, 1, day).unwrap())
        .collect();
    let prices = Series::try_new(dates, vec![3., 2., 3., 1., 3., 6.]).unwrap();
    for (label, result) in [
        (
            "moving average",
            moving_average(&prices, Window::Observations { size: 2, ramp: 0 }).unwrap(),
        ),
        (
            "2 calendar days",
            moving_average(&prices, Window::Days { size: 2, ramp: 2 }).unwrap(),
        ),
        (
            "simple returns",
            returns(&prices, 1, ReturnKind::Simple).unwrap(),
        ),
    ] {
        println!("{label}");
        for (date, value) in result.dates().iter().zip(result.values()) {
            println!("{date} {value:.8}");
        }
    }
    Ok(())
}
