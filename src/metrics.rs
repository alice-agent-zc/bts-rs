//! Performance metrics for backtesting.
//!
//! This module provides tools to calculate:
//! - Max drawdown
//! - Profit factor
//! - Sharpe ratio
//! - Win rate
//!
//! Events generated during backtesting.
//!
//! This module defines the `Event` enum, which represents actions and state changes
//! during a backtest, such as order execution, position updates, and wallet changes.
//! 
//! It needs to enable `metrics` feature to use it. Take a look at [trailing stop](https://github.com/raonagos/bts-rs/blob/master/examples/trailing_stop.rs#L62) for example.

use std::fmt;

use crate::engine::*;

use chrono::{DateTime, Utc};

/// Events generated during a backtest.
///
/// Each event corresponds to an action or state change, such as:
/// - Adding or removing orders/positions.
/// - Updating the wallet balance.
/// - Charging fees.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// An order has been added to the backtest.
    ///
    /// This event is triggered when a new order is created and added to the order queue.
    AddOrder(DateTime<Utc>, Order),

    /// An order has been removed from the backtest.
    ///
    /// This event is triggered when an order is canceled or executed.
    DelOrder(DateTime<Utc>, Order),

    /// A position has been opened.
    ///
    /// This event is triggered when an order is executed and a new position is created.
    AddPosition(DateTime<Utc>, Position),

    /// A position has been closed.
    ///
    /// This event is triggered when a position is closed, either manually or by an exit rule.
    DelPosition(DateTime<Utc>, Position),

    /// The wallet balance has been updated.
    ///
    /// Emitted once per candle, at the bar close, by [`crate::Backtest::run`] — not
    /// once per trade. A per-trade sample sees only the bars where something happened,
    /// so an intra-trade excursion is invisible and the open notional reads as a
    /// realised loss; a per-bar sample is the equity curve.
    WalletUpdate {
        /// Moment
        datetime: DateTime<Utc>,
        /// Unrealized profit and loss at this mark.
        pnl: f64,
        /// Total fees paid.
        fees: f64,
        /// Available funds (not locked in open positions).
        free: f64,
        /// Funds locked in open positions.
        locked: f64,
        /// Mark-to-market equity: cash + the entry notional of the positions still
        /// open + the unrealized pnl at this close. **Not** cash: `Wallet::balance()`
        /// is cash, and opening a position moves notional out of it.
        balance: f64,
    },
}

impl From<(DateTime<Utc>, &Wallet)> for Event {
    fn from((datetime, value): (DateTime<Utc>, &Wallet)) -> Self {
        Self::WalletUpdate {
            datetime,
            locked: value.locked(),
            fees: value.fees_paid(),
            balance: value.total_balance(),
            pnl: value.unrealized_pnl(),
            free: value.free_balance().expect("should give the free balance"),
        }
    }
}

/// A collection of trading metrics calculated from a series of events.
///
/// `Metrics` is used to compute and display key performance indicators (KPIs)
/// for a trading strategy, such as max drawdown, profit factor, Sharpe ratio, and win rate.
/// It is typically constructed from a `Backtest` or a list of `Event`s.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metrics {
    pnl: f64,
    fees: f64,
    balance: f64,
    events: Vec<Event>,
    initial_balance: f64,
}

impl From<&Backtest> for Metrics {
    fn from(value: &Backtest) -> Self {
        // `total_balance()` is equity, not cash: a run that ends holding a position has
        // spent cash on notional that is still worth something, and reading cash here
        // reported 920.00 for an account worth 1020.00 (M5)
        let equity = value.total_balance();
        Self {
            fees: value.fees_paid(),
            balance: equity,
            pnl: equity - value.initial_balance(),
            initial_balance: value.initial_balance(),
            events: value.events().cloned().collect(),
        }
    }
}

impl Metrics {
    /// Creates a new `Metrics` instance from a list of events, an initial balance, a cumulative pnl and a cumulative fees paid.
    pub fn new(events: Vec<Event>, initial_balance: f64, balance: f64, pnl: f64, fees: f64) -> Self {
        Self {
            pnl,
            fees,
            events,
            balance,
            initial_balance,
        }
    }

    /// Returns the initial balance.
    pub fn initial_balance(&self) -> f64 {
        self.initial_balance
    }

    /// Returns the events.
    pub fn events(&self) -> std::slice::Iter<'_, Event> {
        self.events.iter()
    }

    /// Returns the balance — **equity** at the last mark, i.e. cash + the entry notional
    /// of any position still open + its unrealized P&L, minus everything paid in fees.
    ///
    /// This is the number to rank on. `Wallet::balance()` is cash and moves when a
    /// position opens; this does not.
    pub fn balance(&self) -> f64 {
        self.balance
    }

    /// Returns the cumulative fees paid.
    pub fn fees(&self) -> f64 {
        self.fees
    }

    /// Returns the profit and loss: **equity − initial balance**, fees included.
    ///
    /// Not `unrealized_pnl`, which is zero on a closed book and is only the mark on
    /// whatever is still open.
    pub fn pnl(&self) -> f64 {
        self.pnl
    }

    /// Returns the per-bar equity samples the risk metrics are computed from.
    fn equity_curve(&self) -> Vec<f64> {
        let mut curve = Vec::with_capacity(self.events.len());
        for event in &self.events {
            if let Event::WalletUpdate { balance, .. } = event {
                curve.push(*balance);
            }
        }
        curve
    }

    /// Returns the simple per-bar returns of the equity curve.
    fn returns(&self) -> Vec<f64> {
        let mut previous = self.initial_balance;
        let mut returns = Vec::new();
        for balance in self.equity_curve() {
            if previous > 0.0 {
                returns.push((balance - previous) / previous);
            }
            previous = balance;
        }
        returns
    }

    /// Returns the number of bars per year implied by the spacing of the equity
    /// samples, or `None` when the stamps cannot tell (all identical, or fewer than
    /// two of them) — in which case there is nothing to annualise to.
    ///
    /// Read off the data, not passed in: the same run scored at 4h and at 1h bars
    /// annualises to the same number without the caller knowing the timeframe.
    pub fn periods_per_year(&self) -> Option<f64> {
        let mut stamps = Vec::new();
        for event in &self.events {
            if let Event::WalletUpdate { datetime, .. } = event {
                stamps.push(*datetime);
            }
        }
        if stamps.len() < 2 {
            return None;
        }
        let mut deltas: Vec<f64> = stamps
            .windows(2)
            .map(|w| (w[1] - w[0]).num_milliseconds() as f64 / 1000.0)
            .filter(|d| *d > 0.0)
            .collect();
        if deltas.len() * 2 < stamps.len() - 1 {
            return None; // most steps do not advance: the spacing is not a bar clock
        }
        deltas.sort_by(f64::total_cmp);
        let median = deltas[deltas.len() / 2];
        if median <= 0.0 {
            return None;
        }
        let year = 365.25 * 24.0 * 3600.0;
        Some(year / median)
    }

    /// Computes the Sharpe ratio **per bar**: mean per-bar return over its standard
    /// deviation, sample standard deviation (n−1), `risk_free_rate` de-annualised by
    /// the bar spacing.
    ///
    /// Returns `0.0` when the ratio is undefined — fewer than two returns, or a zero
    /// standard deviation. A flat run has a Sharpe of zero, not a NaN: there was no
    /// risk and no return, and a NaN poisons every average it enters.
    pub fn sharpe_ratio_per_bar(&self, risk_free_rate: f64) -> f64 {
        let returns = self.returns();
        if returns.len() < 2 {
            return 0.0;
        }
        let n = returns.len() as f64;
        let periods = self.periods_per_year();
        let per_bar_rf = match periods {
            Some(p) if p > 0.0 => risk_free_rate / p,
            _ => 0.0,
        };
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        if !std_dev.is_finite() || std_dev <= f64::EPSILON {
            return 0.0;
        }
        (mean - per_bar_rf) / std_dev
    }

    /// Computes the Sharpe ratio, **annualised**: the per-bar ratio times
    /// `sqrt(periods_per_year)`, with the periods read off the bar spacing of the run.
    ///
    /// This is the number that is comparable across timeframes and across books.
    /// `risk_free_rate` is the annualised risk-free return (0.0 for none).
    pub fn sharpe_ratio(&self, risk_free_rate: f64) -> f64 {
        let per_bar = self.sharpe_ratio_per_bar(risk_free_rate);
        match self.periods_per_year() {
            Some(periods) if periods > 0.0 => per_bar * periods.sqrt(),
            // no clock in the stamps (a hand-built event list): the per-bar number is
            // the honest answer, and the caller is told it is not annualised
            _ => per_bar,
        }
    }

    /// Returns the net P&L of every closed trade: gross `pnl()` less every fee paid
    /// to open and to close it.
    fn trade_nets(&self) -> Vec<f64> {
        let mut nets = Vec::new();
        for event in &self.events {
            if let Event::DelPosition(_, position) = event {
                if let Ok(net) = position.net_pnl() {
                    nets.push(net);
                }
            }
        }
        nets
    }

    /// How close to zero a net trade P&L has to be before it counts as flat.
    ///
    /// A break-even trade comes out of the arithmetic at ~1e-13, not at 0.0, and a
    /// 1e-13 "gain" against no losses would still print an infinite profit factor.
    fn zero_tolerance(&self) -> f64 {
        self.initial_balance.abs() * 1e-9
    }

    /// Computes the profit factor on **net** trade P&L: gross gains over gross losses,
    /// after fees.
    ///
    /// `0.0` when no trade closed, or when every closed trade nets exactly zero — a
    /// break-even trade is not an infinite profit factor. `INFINITY` only when there
    /// are gains and no losses at all, which is a real (if unrepeatable) shape.
    pub fn profit_factor(&self) -> f64 {
        let mut total_gains = 0.0;
        let mut total_losses = 0.0;
        let eps = self.zero_tolerance();

        for net in self.trade_nets() {
            if net > eps {
                total_gains += net;
            } else if net < -eps {
                total_losses += -net;
            }
        }

        if total_gains == 0.0 {
            return 0.0; // no trade, or nothing won
        }
        if total_losses == 0.0 {
            return f64::INFINITY;
        }

        total_gains / total_losses
    }

    /// Computes the win rate as a percentage of trades that **netted** a profit after
    /// fees. A trade whose gross gain is exactly eaten by its costs is not a win.
    pub fn win_rate(&self) -> f64 {
        let nets = self.trade_nets();
        if nets.is_empty() {
            return 0.0;
        }
        let eps = self.zero_tolerance();
        let wins = nets.iter().filter(|net| **net > eps).count();
        (wins as f64 / nets.len() as f64) * 100.0
    }

    /// Computes the maximum drawdown as a percentage.
    pub fn max_drawdown(&self) -> f64 {
        let mut balance_history = Vec::new();

        for event in &self.events {
            if let Event::WalletUpdate { balance, .. } = event {
                balance_history.push(*balance);
            }
        }

        let mut max_peak = self.initial_balance;
        let mut max_drawdown = 0.0;

        for &balance in &balance_history {
            if balance > max_peak {
                max_peak = balance;
            }
            let drawdown = (max_peak - balance) / max_peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown * 100.0
    }

}

impl fmt::Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Backtest Metrics ===")?;
        writeln!(f, "Initial Balance: {:.2}", self.initial_balance)?;
        writeln!(f, "Final Balance: {:.2}", self.balance)?;
        writeln!(f, "Profit & Loss (P&L): {:.2}", self.pnl)?;
        writeln!(f, "Fees paid: {:.2}", self.fees)?;
        #[allow(clippy::writeln_empty_string)]
        writeln!(f, "")?;
        writeln!(f, "Max Drawdown: {:.2}%", self.max_drawdown())?;
        writeln!(f, "Profit Factor: {:.2}", self.profit_factor())?;
        match self.periods_per_year() {
            Some(periods) => writeln!(
                f,
                "Sharpe Ratio (annualised, {:.1} bars/yr, risk-free rate = 0.0): {:.2}",
                periods,
                self.sharpe_ratio(0.0)
            )?,
            None => writeln!(
                f,
                "Sharpe Ratio (per bar, no bar clock in the stamps, risk-free rate = 0.0): {:.4}",
                self.sharpe_ratio(0.0)
            )?,
        }
        writeln!(f, "Win Rate: {:.2}%", self.win_rate())
    }
}

#[cfg(test)]
// Helper function to create a simple position for testing
fn create_position(pnl: f64) -> Position {
    let order: Order = (OrderType::Market(100.0), 1.0, OrderSide::Buy).into();
    let mut position = Position::from(order);
    // Mock the pnl() method for testing
    // In a real scenario, you would set the exit price or mock the behavior
    // Here, we assume Position has a method to set pnl directly for testing
    position.set_exit_price(100.0 + pnl).unwrap(); // Simulate a P&L of `pnl`
    position
}

#[cfg(test)]
#[test]
fn max_drawdown() {
    let events = vec![
        Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: 10000.0,
            locked: 0.0,
            balance: 10000.0,
        },
        Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: 12000.0,
            locked: 0.0,
            balance: 12000.0,
        },
        Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: 9000.0,
            locked: 0.0,
            balance: 9000.0,
        },
        Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: 11000.0,
            locked: 0.0,
            balance: 11000.0,
        },
    ];
    let metrics = Metrics::new(events, 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.max_drawdown(), 25.0); // (12000 - 9000) / 12000 = 25%
}

#[cfg(test)]
#[test]
fn max_drawdown_no_events() {
    let metrics = Metrics::new(vec![], 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.max_drawdown(), 0.0); // No drawdown if no events
}

#[cfg(test)]
#[test]
fn profit_factor() {
    let winning_position = create_position(20.0);
    let losing_position = create_position(-10.0);
    let events = vec![
        Event::DelPosition(DateTime::default(), winning_position),
        Event::DelPosition(DateTime::default(), losing_position),
    ];
    let metrics = Metrics::new(events, 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.profit_factor(), 2.0); // 20 / 10 = 2.0
}

#[cfg(test)]
#[test]
fn profit_factor_no_losses() {
    let winning_position = create_position(20.0);
    let events = vec![Event::DelPosition(DateTime::default(), winning_position)];
    let metrics = Metrics::new(events, 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.profit_factor(), f64::INFINITY); // No losses
}

#[cfg(test)]
#[test]
fn profit_factor_no_trades() {
    let metrics = Metrics::new(vec![], 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.profit_factor(), 0.0); // no gains to divide: 0.0, not inf
}

#[cfg(test)]
#[test]
fn sharpe_ratio() {
    let events = vec![
        Event::WalletUpdate {
            datetime: DateTime::default(),

            pnl: 0.0,
            fees: 0.0,
            free: 10000.0,
            locked: 0.0,
            balance: 10000.0,
        },
        Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: 10500.0,
            locked: 0.0,
            balance: 10500.0,
        },
        Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: 10300.0,
            locked: 0.0,
            balance: 10300.0,
        },
        Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: 10700.0,
            locked: 0.0,
            balance: 10700.0,
        },
    ];
    let metrics = Metrics::new(events, 10000.0, 0.0, 0.0, 0.0);
    let sharpe = metrics.sharpe_ratio(0.0);
    // Approximate value, since Sharpe ratio depends on standard deviation
    assert!(sharpe > 0.0 && sharpe < 1.0);
}

#[cfg(test)]
#[test]
fn sharpe_ratio_no_events() {
    let metrics = Metrics::new(vec![], 10000.0, 0.0, 0.0, 0.0);
    // no returns -> no ratio. 0.0, not NaN: a NaN poisons every average it enters
    assert_eq!(metrics.sharpe_ratio(0.0), 0.0);
}

#[cfg(test)]
#[test]
fn sharpe_ratio_flat_book_is_zero() {
    let flat = [10000.0, 10000.0, 10000.0, 10000.0];
    let events: Vec<Event> = flat
        .iter()
        .map(|b| Event::WalletUpdate {
            datetime: DateTime::default(),
            pnl: 0.0,
            fees: 0.0,
            free: *b,
            locked: 0.0,
            balance: *b,
        })
        .collect();
    let metrics = Metrics::new(events, 10000.0, 10000.0, 0.0, 0.0);
    assert_eq!(metrics.sharpe_ratio(0.0), 0.0); // zero std: undefined, reported as 0.0
}

#[cfg(test)]
#[test]
fn win_rate() {
    let winning_position = create_position(20.0);
    let losing_position = create_position(-10.0);
    let events = vec![
        Event::DelPosition(DateTime::default(), winning_position),
        Event::DelPosition(DateTime::default(), losing_position),
    ];
    let metrics = Metrics::new(events, 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.win_rate(), 50.0); // 1 win out of 2 trades
}

#[cfg(test)]
#[test]
fn win_rate_no_trades() {
    let metrics = Metrics::new(vec![], 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.win_rate(), 0.0); // No trades
}

#[cfg(test)]
#[test]
fn win_rate_all_winning() {
    let winning_position = create_position(20.0);
    let events = vec![Event::DelPosition(DateTime::default(), winning_position)];
    let metrics = Metrics::new(events, 10000.0, 0.0, 0.0, 0.0);
    assert_eq!(metrics.win_rate(), 100.0); // 1 win out of 1 trade
}

// ---- M1: the drawdown must be read off a per-bar mark-to-market equity curve ----
//
// `run` samples the wallet once per bar; `WalletUpdate.balance` is equity
// (cash + entry notional of the open positions + unrealized at that close). The
// cases below are the audit's: the same 90%-notional position on a +9% path, a flat
// path and a −45% path. Before the fix all three reported 90.00% — the position size,
// not the loss — because the sample was taken per trade, when cash had already paid
// the notional out.

#[cfg(test)]
/// 1000.0 account, `qty` bought at 100 on bar 0 and held to the last bar.
fn hold_through(closes: &[f64], qty: f64) -> Backtest {
    use chrono::{Duration, TimeZone};

    let candles: Vec<Candle> = closes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let t = Utc.timestamp_opt(1_600_000_000 + i as i64 * 3600, 0).unwrap();
            CandleBuilder::builder()
                .open(c)
                .high(c.max(100.0))
                .low(c.min(100.0))
                .close(c)
                .volume(1.0)
                .bid(0.0)
                .open_time(t)
                .close_time(t + Duration::hours(1))
                .build()
                .unwrap()
        })
        .collect();

    let mut bt = Backtest::new(std::sync::Arc::from(candles), 1000.0, None).unwrap();
    let mut i = 0;
    bt.run(|bt, candle| {
        if i == 0 {
            bt.place_order(candle, (OrderType::Market(100.0), qty, OrderSide::Buy).into())?;
        }
        i += 1;
        Ok(())
    })
    .unwrap();
    bt
}

#[cfg(test)]
#[test]
fn max_drawdown_marks_a_held_position_to_market() {
    // 900 of notional on a 1000 account = 90%; price 100 -> 50 is −45% of the account.
    let bt = hold_through(&[100.0, 100.0, 100.0, 50.0], 9.0);
    let m = Metrics::from(&bt);
    assert_eq!(bt.total_balance(), 550.0);
    assert!((m.max_drawdown() - 45.0).abs() < 1e-9, "dd {}", m.max_drawdown());
}

#[cfg(test)]
#[test]
fn max_drawdown_flat_path_with_a_held_position_is_zero() {
    let bt = hold_through(&[100.0; 5], 9.0);
    let m = Metrics::from(&bt);
    assert_eq!(bt.total_balance(), 1000.0);
    assert_eq!(m.max_drawdown(), 0.0);
}

#[cfg(test)]
#[test]
fn max_drawdown_rising_path_with_a_held_position_is_zero() {
    let bt = hold_through(&[100.0, 105.0, 110.0], 9.0);
    let m = Metrics::from(&bt);
    assert_eq!(bt.total_balance(), 1090.0);
    assert_eq!(m.max_drawdown(), 0.0);
}

#[cfg(test)]
#[test]
fn wallet_equity_values_the_open_position() {
    let mut wallet = Wallet::new(1000.0).unwrap();
    wallet.lock(900.0).unwrap();
    wallet.sub(900.0).unwrap(); // cash 100, notional 900 held
    wallet.add_open_cost(900.0);
    assert_eq!(wallet.balance(), 100.0);
    assert_eq!(wallet.equity(), 1000.0);

    wallet.set_unrealized_pnl(-450.0); // the −45% mark
    assert_eq!(wallet.equity(), 550.0);

    wallet.sub_open_cost(900.0); // closed: the notional came back as cash
    wallet.set_unrealized_pnl(0.0);
    assert_eq!(wallet.balance(), 100.0);
    assert_eq!(wallet.equity(), 100.0);
}

// ---- M2/M3/M4/M5: pnl, Sharpe, net PF/WR, balance = equity ----
//
// The audit's cases, on the same 1000.00 account. `round_trip` buys `qty` at the first
// close and closes everything at `exit` on the last bar; fees are 0.1% a side, the
// audit's number.

#[cfg(test)]
fn round_trip(closes: &[f64], qty: f64, exit: f64, fees: bool) -> Backtest {
    use chrono::{Duration, TimeZone};

    let candles: Vec<Candle> = closes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let t = Utc.timestamp_opt(1_600_000_000 + i as i64 * 3600, 0).unwrap();
            CandleBuilder::builder()
                .open(c)
                .high(c)
                .low(c)
                .close(c)
                .volume(1.0)
                .bid(0.0)
                .open_time(t)
                .close_time(t + Duration::hours(1))
                .build()
                .unwrap()
        })
        .collect();
    let n = candles.len();

    let fees = if fees { Some((0.1, 0.1)) } else { None };
    let mut bt = Backtest::new(std::sync::Arc::from(candles), 1000.0, fees).unwrap();
    let mut i = 0;
    bt.run(|bt, candle| {
        if i == 0 {
            bt.place_order(candle, (OrderType::Market(closes[0]), qty, OrderSide::Buy).into())?;
        }
        if i + 1 == n {
            let last = bt.positions().last().copied();
            if let Some(p) = last {
                bt.close_position(candle, &p, exit, true)?;
            }
        }
        i += 1;
        Ok(())
    })
    .unwrap();
    bt
}

#[cfg(test)]
#[test]
fn pnl_is_equity_minus_initial_on_a_closed_book() {
    let bt = round_trip(&[100.0, 105.0, 110.0], 9.0, 110.0, false);
    let m = Metrics::from(&bt);
    assert_eq!(bt.total_balance(), 1090.0);
    assert_eq!(m.pnl(), 90.0); // was 0.00: it read unrealized_pnl of a closed book
    assert_eq!(m.balance(), 1090.0);
}

#[cfg(test)]
#[test]
fn balance_is_equity_while_a_position_is_open() {
    // 9 @ 100 held to 110, never closed: cash 100, equity 1090
    let bt = hold_through(&[100.0, 105.0, 110.0], 9.0);
    let m = Metrics::from(&bt);
    assert_eq!(bt.balance(), 100.0); // cash, by definition
    assert_eq!(m.balance(), 1090.0); // was 100.00: it read cash (M5)
    assert_eq!(m.pnl(), 90.0);
}

#[cfg(test)]
#[test]
fn sharpe_is_annualised_off_the_bar_clock() {
    let bt = round_trip(&[100.0, 105.0, 110.0], 9.0, 110.0, false);
    let m = Metrics::from(&bt);
    let periods = m.periods_per_year().expect("hourly stamps");
    assert!((periods - 365.25 * 24.0).abs() < 1.0, "periods {periods}");
    let ratio = m.sharpe_ratio(0.0) / m.sharpe_ratio_per_bar(0.0);
    assert!((ratio - periods.sqrt()).abs() < 1e-9, "ratio {ratio}");
}

#[cfg(test)]
#[test]
fn sharpe_is_size_invariant() {
    // the same +10% path at 9% and at 90% notional: same returns, same Sharpe
    let small = round_trip(&[100.0, 105.0, 110.0], 0.9, 110.0, false);
    let large = round_trip(&[100.0, 105.0, 110.0], 9.0, 110.0, false);
    let (a, b) = (Metrics::from(&small), Metrics::from(&large));
    let (a, b) = (a.sharpe_ratio(0.0), b.sharpe_ratio(0.0));
    // 10× the notional, 10× the returns: the ratio may only move by compounding noise
    assert!((a - b).abs() / a.abs() < 1e-2, "{a} vs {b}");
}

#[cfg(test)]
#[test]
fn profit_factor_and_win_rate_are_net_of_fees() {
    // gross flat at 100.0, 1.80 of fees: a loss of 1.80, so no gains
    let bt = round_trip(&[100.0, 101.0, 100.0], 9.0, 100.0, true);
    let m = Metrics::from(&bt);
    assert!((m.fees() - 1.8).abs() < 1e-9);
    assert!((m.pnl() + 1.8).abs() < 1e-9);
    assert_eq!(m.profit_factor(), 0.0); // was inf on a gross-flat trade
    assert_eq!(m.win_rate(), 0.0);
}

#[cfg(test)]
#[test]
fn break_even_after_fees_is_not_an_infinite_profit_factor() {
    // gross +1.80 exactly eats 1.80 of fees: the trade is flat, so PF is 0.0, not inf
    let bt = round_trip(&[100.0, 101.0, 100.2], 9.0, 100.2, true);
    let m = Metrics::from(&bt);
    assert!((m.fees() - 1.8).abs() < 1e-9);
    assert!(m.pnl().abs() < 1e-9);
    assert_eq!(m.profit_factor(), 0.0);
    assert_eq!(m.win_rate(), 0.0);
}

#[cfg(test)]
#[test]
fn a_net_winner_still_scores_as_one() {
    let bt = round_trip(&[100.0, 105.0, 110.0], 9.0, 110.0, true);
    let m = Metrics::from(&bt);
    assert!((m.pnl() - (90.0 - 1.8)).abs() < 1e-9, "pnl {}", m.pnl());
    assert_eq!(m.win_rate(), 100.0);
    assert!(m.profit_factor().is_infinite()); // gains, no losses
}
