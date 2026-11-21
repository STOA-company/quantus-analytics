use quantus_math::{moving_average, returns, Date, ReturnKind, Series, Window};
use std::io::{self, Read};
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut lines = input.lines();
    let args: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    let mut dates = Vec::new();
    let mut values = Vec::new();
    for line in lines {
        let mut parts = line.split_whitespace();
        let date: Vec<_> = parts
            .next()
            .unwrap()
            .split('-')
            .map(|v| v.parse::<i32>().unwrap())
            .collect();
        dates.push(Date::new(date[0], date[1] as u8, date[2] as u8).unwrap());
        values.push(parts.next().unwrap().parse::<f64>().unwrap());
    }
    let series = Series::try_new(dates, values).unwrap();
    let result = match args[0] {
        "expanding" => moving_average(&series, Window::Expanding),
        "observations" => moving_average(
            &series,
            Window::Observations {
                size: args[1].parse().unwrap(),
                ramp: args[2].parse().unwrap(),
            },
        ),
        "days" => moving_average(
            &series,
            Window::Days {
                size: args[1].parse().unwrap(),
                ramp: args[2].parse().unwrap(),
            },
        ),
        "returns" => returns(
            &series,
            args[1].parse().unwrap(),
            match args[2] {
                "simple" => ReturnKind::Simple,
                "logarithmic" => ReturnKind::Logarithmic,
                "absolute" => ReturnKind::Absolute,
                _ => panic!("unknown kind"),
            },
        ),
        _ => panic!("unknown operation"),
    };
    match result {
        Ok(result) => {
            for (date, value) in result.dates().iter().zip(result.values()) {
                println!("{date} {value:?}");
            }
        }
        Err(err) => println!("ERROR {err:?}"),
    }
}
