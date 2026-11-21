use quantus_math::{
    beta, correlation, exponential_moving_average, max_drawdown, moving_average, returns,
    volatility, Date, ReturnKind, Series, Window,
};
use std::io::{self, Read};

fn parse_rows<'a>(lines: impl Iterator<Item = &'a str>) -> Series {
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
    Series::try_new(dates, values).unwrap()
}

fn parse_window(kind: &str, size: &str, ramp: &str) -> Window {
    match kind {
        "expanding" => Window::Expanding,
        "observations" => Window::Observations {
            size: size.parse().unwrap(),
            ramp: ramp.parse().unwrap(),
        },
        "days" => Window::Days {
            size: size.parse().unwrap(),
            ramp: ramp.parse().unwrap(),
        },
        _ => panic!("unknown window"),
    }
}

fn parse_return_kind(kind: &str) -> ReturnKind {
    match kind {
        "simple" => ReturnKind::Simple,
        "logarithmic" => ReturnKind::Logarithmic,
        "absolute" => ReturnKind::Absolute,
        _ => panic!("unknown kind"),
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut lines = input.lines();
    let args: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    let result = if args[0] == "correlation" {
        let window = parse_window(args[1], args[2], args[3]);
        let returns_kind = match (args[4], args[5]) {
            ("none", "none") => None,
            (kx, ky) => Some((parse_return_kind(kx), parse_return_kind(ky))),
        };
        let assume_zero_mean = match args[6] {
            "false" => false,
            "true" => true,
            _ => panic!("unknown zero-mean flag"),
        };
        let len_x: usize = args[7].parse().unwrap();
        let remaining: Vec<_> = lines.collect();
        let x = parse_rows(remaining[..len_x].iter().copied());
        let y = parse_rows(remaining[len_x..].iter().copied());
        correlation(&x, &y, window, returns_kind, assume_zero_mean)
    } else if args[0] == "beta" {
        let window = parse_window(args[1], args[2], args[3]);
        let prices = match args[4] {
            "false" => false,
            "true" => true,
            _ => panic!("unknown prices flag"),
        };
        let len_x: usize = args[5].parse().unwrap();
        let remaining: Vec<_> = lines.collect();
        let x = parse_rows(remaining[..len_x].iter().copied());
        let benchmark = parse_rows(remaining[len_x..].iter().copied());
        beta(&x, &benchmark, window, prices)
    } else {
        let series = parse_rows(lines);
        match args[0] {
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
                parse_return_kind(args[2]),
            ),
            "volatility" => {
                let window = parse_window(args[1], args[2], args[3]);
                let returns_kind = match args[4] {
                    "none" => None,
                    k => Some(parse_return_kind(k)),
                };
                let annualization_factor = match args[5] {
                    "auto" => None,
                    v => Some(v.parse().unwrap()),
                };
                let assume_zero_mean = match args[6] {
                    "false" => false,
                    "true" => true,
                    _ => panic!("unknown zero-mean flag"),
                };
                volatility(
                    &series,
                    window,
                    returns_kind,
                    annualization_factor,
                    assume_zero_mean,
                )
            }
            "max_drawdown" => max_drawdown(&series, parse_window(args[1], args[2], args[3])),
            "exponential_moving_average" => {
                let beta = match args[1] {
                    "default" => None,
                    v => Some(v.parse().unwrap()),
                };
                exponential_moving_average(&series, beta)
            }
            _ => panic!("unknown operation"),
        }
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
