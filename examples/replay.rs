use quantus_math::{
    aggregate_risk, backtest_basket, basket_series, beta, bollinger_bands, business_day_count,
    business_day_offset, corr_swap_correlation, correlation, cov, day_count_fraction,
    exponential_moving_average, has_feb_29, is_business_day, macd, max_drawdown, moving_average,
    point_sort_order, prices, relative_strength_index, returns, run_predefined_backtest,
    smoothed_moving_average, sort_risk, std as series_std, subtract_risk, vol_swap_volatility,
    volatility, BasketParam, Date, DayCountConvention, HolidayCalendar, PaymentFrequency,
    PredefinedBacktestInput, PredefinedOrder, PredefinedOrderKind, RebalFreq, ReturnKind, RiskRow,
    RiskTable, Roll, RollingLinearRegression, Series, Timestamp, TradeStatus, Window,
};
use std::collections::BTreeMap;
use std::io::{self, Read};

fn parse_date(s: &str) -> Date {
    let parts: Vec<i32> = s.split('-').map(|v| v.parse::<i32>().unwrap()).collect();
    Date::new(parts[0], parts[1] as u8, parts[2] as u8).unwrap()
}

fn parse_timestamp(s: &str) -> Result<Timestamp, quantus_math::Error> {
    let (d_str, t_str) = s.split_once('T').expect("timestamp must contain T");
    let d = parse_date(d_str);
    let t_parts: Vec<u8> = t_str.split(':').map(|v| v.parse::<u8>().unwrap()).collect();
    Timestamp::new(d, t_parts[0], t_parts[1], t_parts[2])
}

fn parse_week_mask(s: &str) -> [bool; 7] {
    let bytes = s.as_bytes();
    assert_eq!(bytes.len(), 7, "week_mask must be 7 chars");
    let mut mask = [false; 7];
    for (i, &b) in bytes.iter().enumerate() {
        mask[i] = match b {
            b'1' => true,
            b'0' => false,
            _ => panic!("invalid week_mask char"),
        };
    }
    mask
}

fn parse_holidays<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<Date> {
    lines
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(parse_date)
        .collect()
}

fn parse_day_count_convention(s: &str) -> DayCountConvention {
    match s {
        "actual_360" => DayCountConvention::Actual360,
        "actual_364" => DayCountConvention::Actual364,
        "actual_36525" => DayCountConvention::Actual36525,
        "actual_365f" => DayCountConvention::Actual365F,
        "actual_365l" => DayCountConvention::Actual365L,
        "one_one" => DayCountConvention::OneOne,
        _ => panic!("unknown day count convention"),
    }
}

fn parse_payment_frequency(s: &str) -> PaymentFrequency {
    match s {
        "daily" => PaymentFrequency::Daily,
        "weekly" => PaymentFrequency::Weekly,
        "semi_monthly" => PaymentFrequency::SemiMonthly,
        "monthly" => PaymentFrequency::Monthly,
        "semi_quarterly" => PaymentFrequency::SemiQuarterly,
        "quarterly" => PaymentFrequency::Quarterly,
        "tri_annually" => PaymentFrequency::TriAnnually,
        "semi_annually" => PaymentFrequency::SemiAnnually,
        "annually" => PaymentFrequency::Annually,
        _ => panic!("unknown payment frequency"),
    }
}

fn parse_roll(s: &str) -> Roll {
    match s {
        "raise" => Roll::Raise,
        "forward" => Roll::Forward,
        "backward" => Roll::Backward,
        "modified_following" => Roll::ModifiedFollowing,
        "modified_preceding" => Roll::ModifiedPreceding,
        _ => panic!("unknown roll"),
    }
}

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

fn parse_rebal_freq(s: &str) -> RebalFreq {
    match s {
        "daily" => RebalFreq::Daily,
        "weekly" => RebalFreq::Weekly,
        "monthly" => RebalFreq::Monthly,
        _ => panic!("unknown rebal_freq"),
    }
}

enum OwnedBasketParam {
    Fixed(f64),
    Series(Series),
}

fn parse_basket_series_block(lines: &[&str], offset: &mut usize) -> Series {
    let header: Vec<_> = lines[*offset].split_whitespace().collect();
    assert_eq!(header[0], "SERIES");
    let len: usize = header[1].parse().unwrap();
    *offset += 1;
    let s = parse_rows(lines[*offset..*offset + len].iter().copied());
    *offset += len;
    s
}

fn parse_basket_param(lines: &[&str], offset: &mut usize) -> OwnedBasketParam {
    let header: Vec<_> = lines[*offset].split_whitespace().collect();
    match header[0] {
        "FIXED" => {
            *offset += 1;
            OwnedBasketParam::Fixed(header[1].parse::<f64>().unwrap())
        }
        "SERIES" => OwnedBasketParam::Series(parse_basket_series_block(lines, offset)),
        _ => panic!("unknown basket param tag"),
    }
}

fn parse_risk_table_block(lines: &[&str], offset: &mut usize) -> RiskTable {
    let header: Vec<_> = lines[*offset].split('\t').collect();
    assert_eq!(header[0], "TABLE");
    let nrows: usize = header[1].parse().unwrap();
    let unit = match header[2] {
        "NONE" => None,
        "SOME" => Some(header.get(3).copied().unwrap_or("").to_string()),
        _ => panic!("unknown unit tag"),
    };
    *offset += 1;
    let mut rows = Vec::with_capacity(nrows);
    for _ in 0..nrows {
        let fields: Vec<_> = lines[*offset].split('\t').collect();
        assert_eq!(fields.len(), 7, "risk row must have 7 tab-separated fields");
        let date = if fields[0] == "-" {
            None
        } else {
            Some(parse_date(fields[0]))
        };
        let time = if fields[1] == "-" {
            None
        } else {
            Some(fields[1].strip_prefix("T:").unwrap().to_string())
        };
        rows.push(RiskRow {
            date,
            time,
            mkt_type: fields[2].to_string(),
            mkt_asset: fields[3].to_string(),
            mkt_class: fields[4].to_string(),
            mkt_point: fields[5].to_string(),
            value: fields[6].parse::<f64>().unwrap(),
        });
        *offset += 1;
    }
    RiskTable { unit, rows }
}

fn print_risk_table(table: &RiskTable) {
    match &table.unit {
        None => println!("UNIT\tNONE"),
        Some(u) => println!("UNIT\tSOME\t{u}"),
    }
    for row in &table.rows {
        let date_s = match row.date {
            None => "-".to_string(),
            Some(d) => d.to_string(),
        };
        let time_s = match &row.time {
            None => "-".to_string(),
            Some(t) => format!("T:{t}"),
        };
        println!(
            "{date_s}\t{time_s}\t{}\t{}\t{}\t{}\t{:?}",
            row.mkt_type, row.mkt_asset, row.mkt_class, row.mkt_point, row.value
        );
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut lines = input.lines();
    let args: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    if args[0] == "predefined_backtest" {
        let start = parse_date(args[1]);
        let end = parse_date(args[2]);
        let mask = parse_week_mask(args[3]);
        let eod_h: u8 = args[4].parse().unwrap();
        let eod_m: u8 = args[5].parse().unwrap();
        let eod_s: u8 = args[6].parse().unwrap();
        let initial_value: f64 = args[7].parse().unwrap();
        let cash_label = args[8].to_string();
        let n_holidays: usize = args[9].parse().unwrap();
        let n_daily: usize = args[10].parse().unwrap();
        let n_intraday: usize = args[11].parse().unwrap();
        let n_orders: usize = args[12].parse().unwrap();

        let remaining: Vec<_> = lines.collect();
        let mut offset = 0usize;
        let holidays: Vec<Date> = remaining[offset..offset + n_holidays]
            .iter()
            .map(|l| parse_date(l.trim()))
            .collect();
        offset += n_holidays;

        let cal = match HolidayCalendar::try_new(mask, holidays) {
            Ok(c) => c,
            Err(err) => {
                println!("ERROR {err:?}");
                return;
            }
        };

        let mut daily_prices = BTreeMap::new();
        for _ in 0..n_daily {
            let hdr: Vec<_> = remaining[offset].split_whitespace().collect();
            assert_eq!(hdr[0], "DAILY");
            let inst = hdr[1].to_string();
            let nrows: usize = hdr[2].parse().unwrap();
            offset += 1;
            let s = parse_rows(remaining[offset..offset + nrows].iter().copied());
            offset += nrows;
            daily_prices.insert(inst, s);
        }

        let mut intraday_prices = BTreeMap::new();
        for _ in 0..n_intraday {
            let hdr: Vec<_> = remaining[offset].split_whitespace().collect();
            assert_eq!(hdr[0], "INTRADAY");
            let inst = hdr[1].to_string();
            let nrows: usize = hdr[2].parse().unwrap();
            offset += 1;
            let mut ticks = Vec::with_capacity(nrows);
            for &row in &remaining[offset..offset + nrows] {
                let mut parts = row.split_whitespace();
                let ts = match parse_timestamp(parts.next().unwrap()) {
                    Ok(t) => t,
                    Err(err) => {
                        println!("ERROR {err:?}");
                        return;
                    }
                };
                let px: f64 = parts.next().unwrap().parse().unwrap();
                ticks.push((ts, px));
            }
            offset += nrows;
            intraday_prices.insert(inst, ticks);
        }

        let mut orders = Vec::with_capacity(n_orders);
        for _ in 0..n_orders {
            let fields: Vec<_> = remaining[offset].split('\t').collect();
            offset += 1;
            let kind_tag = fields[0];
            let instrument = fields[1].to_string();
            let quantity: f64 = fields[2].parse().unwrap();
            let generation_time = match parse_timestamp(fields[3]) {
                Ok(t) => t,
                Err(err) => {
                    println!("ERROR {err:?}");
                    return;
                }
            };
            let (kind, source) = match kind_tag {
                "MOC" => (
                    PredefinedOrderKind::MarketOnClose {
                        execution_date: parse_date(fields[4]),
                    },
                    fields[5].to_string(),
                ),
                "TWAP" => {
                    let s_ts = match parse_timestamp(fields[4]) {
                        Ok(t) => t,
                        Err(err) => {
                            println!("ERROR {err:?}");
                            return;
                        }
                    };
                    let e_ts = match parse_timestamp(fields[5]) {
                        Ok(t) => t,
                        Err(err) => {
                            println!("ERROR {err:?}");
                            return;
                        }
                    };
                    (
                        PredefinedOrderKind::Twap {
                            start: s_ts,
                            end: e_ts,
                        },
                        fields[6].to_string(),
                    )
                }
                "AT_MARKET" => {
                    let ex_ts = match parse_timestamp(fields[4]) {
                        Ok(t) => t,
                        Err(err) => {
                            println!("ERROR {err:?}");
                            return;
                        }
                    };
                    (
                        PredefinedOrderKind::AtMarket {
                            execution_time: ex_ts,
                        },
                        fields[5].to_string(),
                    )
                }
                "COST" => {
                    let ex_ts = match parse_timestamp(fields[4]) {
                        Ok(t) => t,
                        Err(err) => {
                            println!("ERROR {err:?}");
                            return;
                        }
                    };
                    (
                        PredefinedOrderKind::Cost {
                            execution_time: ex_ts,
                        },
                        fields[5].to_string(),
                    )
                }
                _ => panic!("unknown predefined order kind: {kind_tag}"),
            };
            orders.push(PredefinedOrder {
                instrument,
                quantity,
                generation_time,
                kind,
                source,
            });
        }

        let input = PredefinedBacktestInput {
            start,
            end,
            calendar: &cal,
            initial_value,
            eod_valuation_time: (eod_h, eod_m, eod_s),
            cash_label,
            daily_prices,
            intraday_prices,
            orders,
        };

        match run_predefined_backtest(&input) {
            Ok(res) => {
                println!("PERFORMANCE {}", res.performance.dates().len());
                for (d, v) in res.performance.dates().iter().zip(res.performance.values()) {
                    println!("{d} {v:?}");
                }
                println!("SNAPSHOTS {}", res.snapshots.len());
                for snap in &res.snapshots {
                    println!("SNAP {} {}", snap.date, snap.holdings.len());
                    for (inst, &units) in &snap.holdings {
                        let w = snap.weights[inst];
                        println!("{inst}\t{units:?}\t{w:?}");
                    }
                }
                println!("FILLS {}", res.fills.len());
                for f in &res.fills {
                    println!(
                        "{}\t{:?}\t{:?}\t{}\t{}",
                        f.instrument, f.quantity, f.filled_price, f.execution_time, f.source
                    );
                }
                println!("LEDGER {}", res.trade_ledger.len());
                for e in &res.trade_ledger {
                    let close_ts = match e.close_time {
                        Some(t) => t.to_string(),
                        None => "-".to_string(),
                    };
                    let close_px = match e.close_price {
                        Some(p) => format!("{p:?}"),
                        None => "-".to_string(),
                    };
                    let status = match e.status {
                        TradeStatus::Open => "open",
                        TradeStatus::Closed => "closed",
                    };
                    let pnl = match e.trade_pnl {
                        Some(p) => format!("{p:?}"),
                        None => "-".to_string(),
                    };
                    println!(
                        "{}\t{}\t{}\t{:?}\t{}\t{}\t{}",
                        e.instrument, e.open_time, close_ts, e.open_price, close_px, status, pnl
                    );
                }
            }
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
    if args[0] == "point_sort_order" {
        let ref_date = parse_date(args[1]);
        let point = lines.next().unwrap_or("");
        match point_sort_order(point, ref_date) {
            None => println!("NONE"),
            Some(v) => println!("{v:?}"),
        }
        return;
    }
    if args[0] == "sort_risk" {
        let ref_date = parse_date(args[1]);
        let remaining: Vec<_> = lines.collect();
        let mut offset = 0usize;
        let table = parse_risk_table_block(&remaining, &mut offset);
        let sorted = sort_risk(&table, ref_date);
        print_risk_table(&sorted);
        return;
    }
    if args[0] == "aggregate_risk" {
        let ref_date = parse_date(args[1]);
        let threshold = match args[2] {
            "none" => None,
            v => Some(v.parse::<f64>().unwrap()),
        };
        let ntables: usize = args[3].parse().unwrap();
        let remaining: Vec<_> = lines.collect();
        let mut offset = 0usize;
        let mut tables = Vec::with_capacity(ntables);
        for _ in 0..ntables {
            tables.push(parse_risk_table_block(&remaining, &mut offset));
        }
        match aggregate_risk(&tables, threshold, ref_date) {
            Ok(table) => print_risk_table(&table),
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
    if args[0] == "subtract_risk" {
        let ref_date = parse_date(args[1]);
        let remaining: Vec<_> = lines.collect();
        let mut offset = 0usize;
        let left = parse_risk_table_block(&remaining, &mut offset);
        let right = parse_risk_table_block(&remaining, &mut offset);
        match subtract_risk(&left, &right, ref_date) {
            Ok(table) => print_risk_table(&table),
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
    if args[0] == "backtest_basket" || args[0] == "basket_series" {
        let rebal_freq = parse_rebal_freq(args[1]);
        let num_assets: usize = args[2].parse().unwrap();
        let weights_count: Option<usize> = match args[3] {
            "none" => None,
            n => Some(n.parse().unwrap()),
        };
        let costs_count: Option<usize> = match args[4] {
            "none" => None,
            n => Some(n.parse().unwrap()),
        };
        let remaining: Vec<_> = lines.collect();
        let mut offset = 0usize;
        let mut price_series = Vec::with_capacity(num_assets);
        for _ in 0..num_assets {
            price_series.push(parse_basket_series_block(&remaining, &mut offset));
        }
        let owned_weights: Option<Vec<OwnedBasketParam>> = weights_count.map(|nw| {
            (0..nw)
                .map(|_| parse_basket_param(&remaining, &mut offset))
                .collect()
        });
        let owned_costs: Option<Vec<OwnedBasketParam>> = costs_count.map(|nc| {
            (0..nc)
                .map(|_| parse_basket_param(&remaining, &mut offset))
                .collect()
        });
        let borrowed_weights: Option<Vec<BasketParam<'_>>> = owned_weights.as_ref().map(|ws| {
            ws.iter()
                .map(|p| match p {
                    OwnedBasketParam::Fixed(v) => BasketParam::Fixed(*v),
                    OwnedBasketParam::Series(s) => BasketParam::Series(s),
                })
                .collect()
        });
        let borrowed_costs: Option<Vec<BasketParam<'_>>> = owned_costs.as_ref().map(|cs| {
            cs.iter()
                .map(|p| match p {
                    OwnedBasketParam::Fixed(v) => BasketParam::Fixed(*v),
                    OwnedBasketParam::Series(s) => BasketParam::Series(s),
                })
                .collect()
        });
        if args[0] == "backtest_basket" {
            match backtest_basket(
                &price_series,
                borrowed_weights.as_deref(),
                borrowed_costs.as_deref(),
                rebal_freq,
            ) {
                Ok(res) => {
                    println!("NAV");
                    for (date, value) in res.nav.dates().iter().zip(res.nav.values()) {
                        println!("{date} {value:?}");
                    }
                    for (a, ws) in res.actual_weights.iter().enumerate() {
                        println!("WEIGHT {a}");
                        for (date, value) in ws.dates().iter().zip(ws.values()) {
                            println!("{date} {value:?}");
                        }
                    }
                }
                Err(err) => println!("ERROR {err:?}"),
            }
        } else {
            match basket_series(
                &price_series,
                borrowed_weights.as_deref(),
                borrowed_costs.as_deref(),
                rebal_freq,
            ) {
                Ok(nav) => {
                    for (date, value) in nav.dates().iter().zip(nav.values()) {
                        println!("{date} {value:?}");
                    }
                }
                Err(err) => println!("ERROR {err:?}"),
            }
        }
        return;
    }
    if args[0] == "has_feb_29" {
        let start = parse_date(args[1]);
        let end = parse_date(args[2]);
        println!("{}", has_feb_29(start, end));
        return;
    }
    if args[0] == "day_count_fraction" {
        let start = parse_date(args[1]);
        let end = parse_date(args[2]);
        let conv = parse_day_count_convention(args[3]);
        let freq = parse_payment_frequency(args[4]);
        println!("{:?}", day_count_fraction(start, end, conv, freq));
        return;
    }
    if args[0] == "is_business_day" {
        let date = parse_date(args[1]);
        let mask = parse_week_mask(args[2]);
        let holidays = parse_holidays(lines);
        match HolidayCalendar::try_new(mask, holidays) {
            Ok(cal) => println!("{}", is_business_day(date, &cal)),
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
    if args[0] == "business_day_offset" {
        let date = parse_date(args[1]);
        let offset: i32 = args[2].parse().unwrap();
        let roll = parse_roll(args[3]);
        let mask = parse_week_mask(args[4]);
        let holidays = parse_holidays(lines);
        match HolidayCalendar::try_new(mask, holidays)
            .and_then(|cal| business_day_offset(date, offset, roll, &cal))
        {
            Ok(d) => println!("{d}"),
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
    if args[0] == "business_day_count" {
        let begin = parse_date(args[1]);
        let end = parse_date(args[2]);
        let mask = parse_week_mask(args[3]);
        let holidays = parse_holidays(lines);
        match HolidayCalendar::try_new(mask, holidays) {
            Ok(cal) => println!("{}", business_day_count(begin, end, &cal)),
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
    if args[0] == "rolling_ols" {
        let window: usize = args[1].parse().unwrap();
        let fit_intercept = match args[2] {
            "false" => false,
            "true" => true,
            _ => panic!("unknown fit_intercept flag"),
        };
        let n_predictors: usize = args[3].parse().unwrap();
        let pred_lens: Vec<usize> = args[4..4 + n_predictors]
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        let remaining: Vec<_> = lines.collect();
        let mut offset = 0usize;
        let mut predictors = Vec::with_capacity(n_predictors);
        for len in pred_lens {
            predictors.push(parse_rows(remaining[offset..offset + len].iter().copied()));
            offset += len;
        }
        let y = parse_rows(remaining[offset..].iter().copied());
        match RollingLinearRegression::fit(&predictors, &y, window, fit_intercept) {
            Ok(reg) => {
                let cols: Vec<String> =
                    reg.column_indices().iter().map(|c| c.to_string()).collect();
                println!("COLS {}", cols.join(" "));
                for &col in reg.column_indices() {
                    let s = reg.coefficient(col).unwrap();
                    println!("COEF {col}");
                    for (date, value) in s.dates().iter().zip(s.values()) {
                        println!("{date} {value:?}");
                    }
                }
                println!("R_SQUARED");
                for (date, value) in reg.r_squared().dates().iter().zip(reg.r_squared().values()) {
                    println!("{date} {value:?}");
                }
                println!("FITTED");
                for (date, value) in reg
                    .fitted_values()
                    .dates()
                    .iter()
                    .zip(reg.fitted_values().values())
                {
                    println!("{date} {value:?}");
                }
                println!("STD_ERR");
                for (date, value) in reg
                    .standard_deviation_of_errors()
                    .dates()
                    .iter()
                    .zip(reg.standard_deviation_of_errors().values())
                {
                    println!("{date} {value:?}");
                }
            }
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
    if args[0] == "bollinger_bands" {
        let window = parse_window(args[1], args[2], args[3]);
        let k = match args[4] {
            "default" => None,
            v => Some(v.parse().unwrap()),
        };
        let series = parse_rows(lines);
        match bollinger_bands(&series, window, k) {
            Ok(bands) => {
                assert_eq!(bands.lower.dates(), bands.upper.dates());
                for ((date, lower), upper) in bands
                    .lower
                    .dates()
                    .iter()
                    .zip(bands.lower.values())
                    .zip(bands.upper.values())
                {
                    println!("{date} {lower:?} {upper:?}");
                }
            }
            Err(err) => println!("ERROR {err:?}"),
        }
        return;
    }
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
    } else if args[0] == "corr_swap_correlation" {
        let n_days = match args[1] {
            "default" => None,
            v => Some(v.parse().unwrap()),
        };
        let assume_zero_mean = match args[2] {
            "false" => false,
            "true" => true,
            _ => panic!("unknown zero-mean flag"),
        };
        let len_x: usize = args[3].parse().unwrap();
        let remaining: Vec<_> = lines.collect();
        let x = parse_rows(remaining[..len_x].iter().copied());
        let y = parse_rows(remaining[len_x..].iter().copied());
        corr_swap_correlation(&x, &y, n_days, assume_zero_mean)
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
    } else if args[0] == "cov" {
        let window = parse_window(args[1], args[2], args[3]);
        let len_x: usize = args[4].parse().unwrap();
        let remaining: Vec<_> = lines.collect();
        let x = parse_rows(remaining[..len_x].iter().copied());
        let y = parse_rows(remaining[len_x..].iter().copied());
        cov(&x, &y, window)
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
            "prices" => {
                let initial = match args[1] {
                    "default" => None,
                    v => Some(v.parse().unwrap()),
                };
                let kind = match args[2] {
                    "default" => None,
                    k => Some(parse_return_kind(k)),
                };
                prices(&series, initial, kind)
            }
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
            "vol_swap_volatility" => {
                let n_days = match args[1] {
                    "default" => None,
                    v => Some(v.parse().unwrap()),
                };
                let annualization_factor = match args[2] {
                    "default" => None,
                    v => Some(v.parse().unwrap()),
                };
                let assume_zero_mean = match args[3] {
                    "false" => false,
                    "true" => true,
                    _ => panic!("unknown zero-mean flag"),
                };
                vol_swap_volatility(&series, n_days, annualization_factor, assume_zero_mean)
            }
            "max_drawdown" => max_drawdown(&series, parse_window(args[1], args[2], args[3])),
            "exponential_moving_average" => {
                let beta = match args[1] {
                    "default" => None,
                    v => Some(v.parse().unwrap()),
                };
                exponential_moving_average(&series, beta)
            }
            "smoothed_moving_average" => {
                smoothed_moving_average(&series, parse_window(args[1], args[2], args[3]))
            }
            "std" => series_std(&series, parse_window(args[1], args[2], args[3])),
            "relative_strength_index" => {
                let window = if args[1] == "default" {
                    None
                } else {
                    Some(parse_window(args[1], args[2], args[3]))
                };
                relative_strength_index(&series, window)
            }
            "macd" => {
                let parse_opt_span = |s: &str| match s {
                    "default" => None,
                    v => Some(v.parse().unwrap()),
                };
                macd(
                    &series,
                    parse_opt_span(args[1]),
                    parse_opt_span(args[2]),
                    parse_opt_span(args[3]),
                )
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
