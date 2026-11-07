//! Simulator-only extension of a sliding-window HSL capture.
//!
//! Once a scope's current episode is older than the configured lookback, the
//! clipped observation changes every minute (its start moves), so no cached
//! controller state is valid and the shared evaluator rebuilds the whole
//! window: minute tape, history reconstruction, trace and replay. Those rebuilt
//! facts are, minute to minute, the previous facts minus the expired left edge
//! plus one new sample, provided the selected fills, the position and the
//! candle tape are unchanged. This cache keeps the composed trace of the last
//! full reconstruction and feeds the identical clipped trace to the unchanged
//! controller replay. Permission is never reused; any changed input returns to
//! the full reconstruction, which refreshes or discards the cache.
use super::hsl_runtime::{Scope, SignalSettings};
use super::*;
use crate::hsl_controller as controller;
use crate::hsl_evaluator as evaluator;
use crate::hsl_history::{Position, PositionSide};
use crate::hsl_snapshot::Mode;
use std::collections::BTreeSet;

pub(super) type Windows = std::collections::BTreeMap<(Option<usize>, Option<usize>), Window>;

pub(super) struct Window {
    now: i64,
    start: i64,
    fill_range: (usize, usize),
    size_bits: u64,
    basis_bits: u64,
    /// Close the previous capture valued its current sample with.
    mark: f64,
    /// The composed trace; only its first episode expires and only its last
    /// episode grows while the retained fills are unchanged.
    episodes: Vec<controller::Episode>,
    /// Quantity, basis and centered realized cashflow a sample after the last
    /// retained fill carries.
    tail_size: f64,
    tail_basis: f64,
    tail_realized: f64,
    reasons: BTreeSet<String>,
    /// Samples appended since the last full reconstruction (test coverage only).
    #[cfg(test)]
    extended: u32,
}

impl Backtest<'_> {
    /// Keep the facts of a full coin reconstruction when they form one plain
    /// exposed episode on a complete minute tape; otherwise forget the scope.
    pub(super) fn remember_hsl_window(
        &mut self,
        k: usize,
        key: (Option<usize>, Option<usize>),
        capture: (i64, i64, (usize, usize), bool),
        facts: Option<evaluator::SimulatorFacts>,
    ) {
        self.hsl_windows.remove(&key);
        let (Some(side), Some(coin)) = key else {
            return;
        };
        let (now, start, fill_range, tape_complete) = capture;
        let Some(facts) = facts else { return };
        let position = if side == LONG {
            self.positions.long[coin]
        } else {
            self.positions.short[coin]
        };
        let exposed = position.size != 0.0;
        // History after the last retained fill must value the same inventory the
        // current sample does: a reconstructed tail that disagrees with the
        // current position (estimated opening or a flat reconciliation) is not
        // extended here.
        if !tape_complete
            || !facts.single_pair
            || facts.episodes.is_empty()
            || facts.boundaries + 1 != facts.episodes.len()
            || (facts.tail_size != 0.0) != exposed
            || !self.coin_is_valid_at(coin, k)
        {
            return;
        }
        let first = facts.episodes.first().unwrap();
        let last = facts.episodes.last().unwrap();
        if facts.episodes.iter().any(|e| {
            e.entry_reference.is_some()
                || e.points
                    .iter()
                    .any(|p| p.cashflow_reference_delta.is_some())
        }) || first.points.first().is_none_or(|p| p.timestamp != start)
            || last
                .points
                .last()
                .is_none_or(|p| p.timestamp != now || p.exposed != exposed || p.flatten)
        {
            return;
        }
        let tail_realized = last.points.last().unwrap().pnl;
        self.hsl_windows.insert(
            key,
            Window {
                now,
                start,
                fill_range,
                size_bits: position.size.to_bits(),
                basis_bits: position.price.to_bits(),
                mark: self.hlcvs_value(k, coin, CLOSE),
                episodes: facts.episodes,
                tail_size: facts.tail_size,
                tail_basis: facts.tail_basis,
                tail_realized,
                reasons: facts.reasons,
                #[cfg(test)]
                extended: 0,
            },
        );
    }

    /// Extend the remembered capture by exactly one bar and replay the
    /// controller over the same clipped trace the full reconstruction would
    /// compose. Any difference in the capture inputs defers to that path.
    pub(super) fn advance_hsl_window(
        &mut self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
        settings: SignalSettings,
        cutoff: Option<(i64, usize)>,
    ) -> Result<Option<Scope>, String> {
        let (Some(s), Some(c)) = (side, coin) else {
            return Ok(None);
        };
        let key = (side, coin);
        let Some(window) = self.hsl_windows.get(&key) else {
            return Ok(None);
        };
        if self.interval_ms != 60_000 || k >= self.hlcvs.shape()[0] {
            return Ok(None);
        }
        let now = (self.first_timestamp_ms + (k as u64 + 1) * self.interval_ms) as i64;
        let start =
            now - (self.backtest_params.pnls_max_lookback_days * 86_400_000.0).round() as i64;
        let start = cutoff.map_or(start, |(timestamp, _)| start.max(timestamp));
        if now != window.now + 60_000 || start != window.start + 60_000 {
            return Ok(None);
        }
        let fill_start = self
            .fills
            .partition_point(|f| (f.timestamp_ms as i128) < start as i128)
            .max(cutoff.map_or(0, |(_, consumed)| consumed));
        let fill_end = self
            .fills
            .partition_point(|f| (f.timestamp_ms as i128) <= now as i128);
        if fill_start < window.fill_range.0 || fill_end < window.fill_range.1 {
            return Ok(None);
        }
        // Only this scope's executions change its pair; other scopes' fills may
        // enter or leave the window freely.
        let name = &self.backtest_params.coins[c];
        let selected = |fill: &Fill| fill.coin == *name && fill.order_type.is_long() == (s == LONG);
        if self.fills[window.fill_range.0..fill_start]
            .iter()
            .any(selected)
            || self.fills[window.fill_range.1..fill_end]
                .iter()
                .any(selected)
        {
            return Ok(None);
        }
        let position = if s == LONG {
            self.positions.long[c]
        } else {
            self.positions.short[c]
        };
        if position.size.to_bits() != window.size_bits
            || position.price.to_bits() != window.basis_bits
            || !self.coin_is_valid_at(c, k)
        {
            return Ok(None);
        }
        let mark = self.hlcvs_value(k, c, CLOSE);
        if !mark.is_finite() || mark <= 0.0 {
            return Ok(None);
        }
        let slots = self.hard_stop_coin_slot_n_positions(s) as u64;
        if slots == 0 {
            // The full path reports the inactive scope and refreshes the cache.
            return Ok(None);
        }
        let exposure_budget = self.hsl_exposure_budget(side, coin);
        let Ok(budget) = evaluator::budget(
            self.balance.usd_total_balance,
            &Mode::Coin,
            slots,
            settings.scale_budget_with_excess_allowance,
            exposure_budget.as_ref(),
        ) else {
            return Ok(None);
        };
        let market = &self.exchange_params_list[c];
        let pside = if s == LONG {
            PositionSide::Long
        } else {
            PositionSide::Short
        };
        let direction = if s == LONG { 1.0 } else { -1.0 };
        let valuation = Position {
            size: position.size,
            basis: position.price,
            mark,
            multiplier: market.c_mult,
            quantity_step: Some(market.qty_step),
            inverse: false,
            pside,
        };
        // The previous current sample becomes a historical one: the reference
        // reconstruction values history after the last retained fill with the
        // reconstructed tail inventory at that minute's close.
        let historical =
            valuation.pnl(direction * window.tail_size, window.tail_basis, window.mark);
        let current = valuation.pnl(direction * position.size.abs(), position.price, mark);
        if !historical.is_finite() || !current.is_finite() {
            self.hsl_windows.remove(&key);
            return Ok(None);
        }
        let normalized = |value: f64| if value == 0.0 { 0.0 } else { value };
        let fill_count = self.fills.len();
        let window = self.hsl_windows.get_mut(&key).unwrap();
        let first = window.episodes.first_mut().unwrap();
        let expired = first.points.partition_point(|p| p.timestamp < start);
        first.points.drain(..expired);
        let first_empty = first.points.is_empty();
        let tail_realized = window.tail_realized;
        let previous = window.now;
        let last = window.episodes.last_mut().unwrap();
        if first_empty || last.points.last().is_none_or(|p| p.timestamp != previous) {
            self.hsl_windows.remove(&key);
            return Ok(None);
        }
        last.points.last_mut().unwrap().upnl = normalized(historical);
        last.points.push(controller::Point {
            timestamp: now,
            pnl: tail_realized,
            upnl: normalized(current),
            exposed: position.size != 0.0,
            flatten: false,
            cashflow_reference_delta: None,
        });
        let input = controller::Input {
            episodes: std::mem::take(&mut window.episodes),
            now,
            start,
            budget,
            span: settings.span,
            threshold: settings.threshold,
            cooldown_ms: settings.cooldown_ms,
            restart: settings.restart,
        };
        let replay = controller::replay_latest_with_events(&input);
        window.episodes = input.episodes;
        window.now = now;
        window.start = start;
        window.fill_range = (fill_start, fill_end);
        window.mark = mark;
        #[cfg(test)]
        {
            window.extended += 1;
        }
        let mut reasons = window.reasons.clone();
        let replay = replay?;
        if replay.numeric_range_approximation {
            reasons.insert("numeric_range_approximation".into());
        }
        let decision = replay
            .decisions
            .into_iter()
            .last()
            .ok_or("empty HSL evaluation")?;
        // The seeded shortcut trace is not refreshed here: while the window start
        // keeps moving past the episode start, neither shortcut can run, and the
        // next flat-boundary reconstruction reseeds it.
        let result = evaluator::Output {
            decision: Some(decision),
            events: replay.events,
            reasons,
            observations: replay.observations,
            episodes: window.episodes.len(),
            cursor: replay.cursor,
            simulator: None,
        };
        Ok(Some(Scope {
            timestamp: now,
            fill_count,
            budget,
            slots,
            side,
            coin,
            result,
        }))
    }
}

#[cfg(test)]
mod tests {
    //! The extension must reproduce, bit for bit, the full reconstruction it
    //! replaces. Two simulators receive the same random fills, candles, budget
    //! and slot changes; one forgets its windows before every bar close and so
    //! always takes the full path. Coverage counters prove the cases reach the
    //! window regime, change positions under it, expire fills past the left
    //! edge, keep completed episodes in flat scopes, split a left-truncated
    //! reconstruction at a fill and expire a whole leading episode.
    use super::*;
    use crate::hsl_controller::Action;
    use crate::types::{
        BacktestParams, BotParamsPair, EquityHardStopLossConfig, ExchangeParams, Order, OrderType,
    };
    use ndarray::{Array1, Array3};
    use std::collections::BTreeMap;

    const DAY: u64 = 86_400_000;
    const FIRST: u64 = 10 * DAY;
    const LOOKBACK_BARS: usize = 1440;

    /// SplitMix64: deterministic without a dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn chance(&mut self, p: f64) -> bool {
            self.unit() < p
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
        fn gauss(&mut self) -> f64 {
            (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732
        }
    }

    enum Execution {
        /// Open or add this quantity.
        Increase(f64),
        /// Close this fraction of the held quantity; `0.0` closes everything.
        Reduce(f64),
    }

    struct PlannedFill {
        coin: usize,
        side: usize,
        execution: Execution,
        price: f64,
    }

    #[derive(Default)]
    struct Bar {
        fills: Vec<PlannedFill>,
        /// Whether an emulated protective close executes a pending panic.
        panic_close: bool,
        balance_delta: f64,
        slots: Option<(usize, usize)>,
    }

    struct Scenario {
        rows: usize,
        coins: usize,
        candles: Array3<f64>,
        btc: Array1<f64>,
        params: BacktestParams,
        bots: Vec<BotParamsPair>,
        exchange: Vec<ExchangeParams>,
        bars: Vec<Bar>,
    }

    fn policy(rng: &mut Rng) -> hsl_runtime::Policy {
        hsl_runtime::Policy {
            enabled: true,
            scale_budget_with_excess_allowance: rng.chance(0.3),
            red_threshold: rng.range(0.03, 0.5),
            ema_span_minutes: [1.0, 3.5, 60.0, 308.5, 1440.0][rng.below(5)],
            cooldown_minutes_after_red: [0.0, 5.0, 120.0, 2000.0][rng.below(4)],
            restart_after_red_policy: Some(if rng.chance(0.7) { "always" } else { "never" }.into()),
            panic_close_order_type: "limit".into(),
        }
    }

    fn scenario(seed: u64) -> Scenario {
        let mut rng = Rng(seed);
        let coins = 1 + rng.below(2);
        // Mostly short runs past the lookback, every tenth case a long one.
        let extra = if seed.is_multiple_of(10) { 500 } else { 120 };
        let rows = LOOKBACK_BARS + 40 + rng.below(extra);
        let mut candles = Array3::<f64>::zeros((rows, coins, 4));
        for coin in 0..coins {
            let mut close = 100.0 * (coin + 1) as f64;
            let sigma = rng.range(0.0003, 0.006);
            let mut drift = rng.range(-0.0004, 0.0004);
            for k in 0..rows {
                if rng.chance(1.0 / 400.0) {
                    drift = -drift;
                }
                let open = close;
                close *= (drift + sigma * rng.gauss()).exp();
                candles[[k, coin, HIGH]] = open.max(close) * (1.0 + 0.002 * rng.unit());
                candles[[k, coin, LOW]] = open.min(close) * (1.0 - 0.002 * rng.unit());
                candles[[k, coin, CLOSE]] = close;
                candles[[k, coin, VOLUME]] = 1.0;
            }
        }
        let mut bot = BotParamsPair::default();
        for (side, slots) in [(&mut bot.long, 4), (&mut bot.short, 2)] {
            side.n_positions = slots;
            side.wallet_exposure_limit = rng.range(0.1, 2.0);
            side.total_wallet_exposure_limit = rng.range(0.5, 4.0);
            side.risk_we_excess_allowance_pct = rng.range(0.0, 0.5);
        }
        let mut coin_policies = BTreeMap::new();
        for coin in 0..coins {
            if rng.chance(0.25) {
                coin_policies.insert(format!("C{coin}"), [policy(&mut rng), policy(&mut rng)]);
            }
        }
        let params = BacktestParams {
            starting_balance: 10_000.0,
            maker_fee: 0.0,
            taker_fee: 0.0,
            coins: (0..coins).map(|i| format!("C{i}")).collect(),
            active_coin_indices: None,
            first_timestamp_ms: FIRST,
            requested_start_timestamp_ms: FIRST,
            first_valid_indices: vec![0; coins],
            last_valid_indices: vec![rows - 1; coins],
            warmup_minutes: vec![0; coins],
            trade_start_indices: vec![0; coins],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: false,
            hsl_detailed_report: rng.chance(0.5),
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 1.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig {
                hsl: Some(hsl_runtime::Config {
                    mode: "coin".into(),
                    sides: [policy(&mut rng), policy(&mut rng)],
                    portfolio: None,
                    coins: coin_policies,
                }),
            },
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };
        // Long episodes keep scopes in the window regime; rare adds, partial
        // closes and reopenings change the retained fills under it.
        let qty_unit = [0.1, 1.0, 3.0][rng.below(3)];
        let p_open = 1.0 / rng.range(100.0, 400.0);
        let p_close = 1.0 / rng.range(2000.0, 5000.0);
        let p_add = 1.0 / rng.range(400.0, 1500.0);
        let p_partial = 1.0 / rng.range(600.0, 2000.0);
        let mut bars = Vec::with_capacity(rows);
        for k in 0..rows {
            let mut bar = Bar {
                panic_close: rng.chance(0.8),
                ..Default::default()
            };
            for coin in 0..coins {
                for side in [LONG, SHORT] {
                    let jitter = 1.0 + rng.range(-0.001, 0.001);
                    let price = (candles[[k, coin, CLOSE]] * jitter * 100.0).round() / 100.0;
                    let qty = qty_unit * (1 + rng.below(10)) as f64;
                    // Drawn unconditionally so the plan never depends on state.
                    let (open, add, close, partial) = (
                        rng.chance(if k < 30 { 0.1 } else { p_open }),
                        rng.chance(p_add),
                        rng.chance(p_close),
                        rng.chance(p_partial),
                    );
                    let fraction = if close { 0.0 } else { rng.range(0.2, 0.8) };
                    let execution = if open || add {
                        Execution::Increase(qty)
                    } else if close || partial {
                        Execution::Reduce(fraction)
                    } else {
                        continue;
                    };
                    bar.fills.push(PlannedFill {
                        coin,
                        side,
                        execution,
                        price,
                    });
                }
            }
            if rng.chance(1.0 / 700.0) {
                bar.balance_delta = rng.range(-2000.0, 2000.0);
            }
            if rng.chance(1.0 / 900.0) {
                bar.slots = Some((1 + rng.below(4), 1 + rng.below(2)));
            }
            bars.push(bar);
        }
        Scenario {
            rows,
            coins,
            candles,
            btc: Array1::from_elem(rows, 50_000.0),
            params,
            bots: vec![bot; coins],
            exchange: vec![
                ExchangeParams {
                    c_mult: 1.0,
                    qty_step: 0.1,
                    price_step: 0.01,
                    min_qty: 0.1,
                    ..Default::default()
                };
                coins
            ],
            bars,
        }
    }

    fn position(bt: &Backtest<'_>, side: usize, coin: usize) -> crate::types::Position {
        if side == LONG {
            bt.positions.long[coin]
        } else {
            bt.positions.short[coin]
        }
    }

    fn execute(bt: &mut Backtest<'_>, k: usize, coin: usize, side: usize, qty: f64, price: f64) {
        let exec = OrderFillExecution {
            price,
            fee_rate: 0.0005,
            liquidity: "maker",
        };
        let (order_type, signed) = match (side == LONG, qty > 0.0) {
            (true, true) => (OrderType::EntryInitialNormalLong, qty),
            (true, false) => (OrderType::CloseGridLong, qty),
            (false, true) => (OrderType::EntryInitialNormalShort, -qty),
            (false, false) => (OrderType::CloseGridShort, -qty),
        };
        let order = Order {
            qty: signed,
            price,
            order_type,
        };
        match (side == LONG, qty > 0.0) {
            (true, true) => bt.process_entry_fill_long(k, coin, &order, exec),
            (false, true) => bt.process_entry_fill_short(k, coin, &order, exec),
            (true, false) => bt.process_close_fill_long(k, coin, &order, exec).unwrap(),
            (false, false) => bt.process_close_fill_short(k, coin, &order, exec).unwrap(),
        }
    }

    /// Apply one bar's plan the way the simulator loop would: protective closes
    /// for a pending panic, no entries while a scope is not normal, then the
    /// planned executions, account and slot changes.
    fn apply(bt: &mut Backtest<'_>, k: usize, bar: &Bar, coins: usize) -> bool {
        bt.current_step = k;
        let mut changed = false;
        for coin in 0..coins {
            for side in [LONG, SHORT] {
                let held = position(bt, side, coin).size;
                if held != 0.0 && bar.panic_close && bt.hsl_action(side, coin) == Action::Panic {
                    execute(
                        bt,
                        k,
                        coin,
                        side,
                        -held.abs(),
                        bt.hlcvs_value(k, coin, CLOSE),
                    );
                    changed = true;
                }
            }
        }
        for fill in &bar.fills {
            let held = position(bt, fill.side, fill.coin).size.abs();
            match fill.execution {
                Execution::Increase(qty) => {
                    if bt.hsl_action(fill.side, fill.coin) != Action::Normal {
                        continue;
                    }
                    execute(bt, k, fill.coin, fill.side, qty, fill.price);
                }
                Execution::Reduce(fraction) => {
                    if held == 0.0 {
                        continue;
                    }
                    let mut qty = round_(held * fraction, 0.1);
                    if qty <= 0.0 || qty > held {
                        qty = held;
                    }
                    execute(bt, k, fill.coin, fill.side, -qty, fill.price);
                }
            }
            changed = true;
        }
        if bar.balance_delta != 0.0 {
            let delta = bar
                .balance_delta
                .max(1_000.0 - bt.balance.usd_total_balance);
            bt.balance.usd_total_balance += delta;
            bt.balance.usd_cash_wallet += delta;
        }
        if let Some((long, short)) = bar.slots {
            bt.effective_n_positions.long = long;
            bt.effective_n_positions.short = short;
        }
        changed
    }

    fn signature(bt: &Backtest<'_>) -> String {
        let mut out = String::new();
        for scope in &bt.hsl_scopes {
            let cursor = scope.result.cursor.as_ref().map(|c| {
                (
                    c.peak_delta.to_bits(),
                    c.first_required,
                    c.exposed,
                    c.threshold_sensitive,
                )
            });
            out.push_str(&format!(
                "{:?}/{:?} t={} fills={} budget={} slots={} {} {:?}\n",
                scope.side,
                scope.coin,
                scope.timestamp,
                scope.fill_count,
                scope.budget.to_bits(),
                scope.slots,
                serde_json::to_string(&scope.result).unwrap(),
                cursor
            ));
        }
        for coin in 0..bt.n_coins {
            for side in [LONG, SHORT] {
                let p = position(bt, side, coin);
                out.push_str(&format!(
                    "pos {side}/{coin} {} {}\n",
                    p.size.to_bits(),
                    p.price.to_bits()
                ));
            }
        }
        out.push_str(&format!(
            "balance {} fills {}\n",
            bt.balance.usd_total_balance.to_bits(),
            bt.fills.len()
        ));
        out
    }

    #[derive(Default, Debug)]
    struct Coverage {
        cases: usize,
        cases_with_hits: usize,
        bars: usize,
        hits: usize,
        flat_hits: usize,
        /// Extensions of a trace holding a reconstructed flat boundary.
        multi_episode_hits: usize,
        full_after_hit_position_change: usize,
        full_after_hit_fill_expiry: usize,
        /// A fill under the window regime split the left-truncated
        /// reconstruction into episodes, and the split trace was remembered.
        boundary_after_fill: usize,
        /// The first of several remembered episodes expired past the left edge.
        full_after_episode_expiry: usize,
        panic_decisions: usize,
        halted_decisions: usize,
        restart_events: usize,
        errors: usize,
    }

    impl Coverage {
        fn add(&mut self, other: &Coverage) {
            self.cases += other.cases;
            self.cases_with_hits += other.cases_with_hits;
            self.bars += other.bars;
            self.hits += other.hits;
            self.flat_hits += other.flat_hits;
            self.multi_episode_hits += other.multi_episode_hits;
            self.full_after_hit_position_change += other.full_after_hit_position_change;
            self.full_after_hit_fill_expiry += other.full_after_hit_fill_expiry;
            self.boundary_after_fill += other.boundary_after_fill;
            self.full_after_episode_expiry += other.full_after_episode_expiry;
            self.panic_decisions += other.panic_decisions;
            self.halted_decisions += other.halted_decisions;
            self.restart_events += other.restart_events;
            self.errors += other.errors;
        }
    }

    /// What a remembered window looked like before a bar close.
    #[derive(Clone, Copy)]
    struct Remembered {
        extended: u32,
        episodes: usize,
        /// Last sample of the first episode; once it falls before the window
        /// start, the whole first episode has expired.
        first_episode_end: i64,
    }

    fn run_case(seed: u64, cov: &mut Coverage) {
        let s = scenario(seed);
        let make = || {
            Backtest::new(
                s.candles.view(),
                s.btc.view(),
                s.bots.clone(),
                s.exchange.clone(),
                &s.params,
            )
        };
        let mut cached = make();
        let mut full = make();
        cov.cases += 1;
        let mut case_hits = 0;
        let lookback_ms = LOOKBACK_BARS as u64 * 60_000;
        for k in 0..s.rows {
            let bar = &s.bars[k];
            let changed_cached = apply(&mut cached, k, bar, s.coins);
            let changed_full = apply(&mut full, k, bar, s.coins);
            assert_eq!(changed_cached, changed_full, "seed={seed} k={k}");
            // Which scopes still own an extensible window before this close.
            let before: BTreeMap<_, _> = cached
                .hsl_windows
                .iter()
                .map(|(key, w)| {
                    let remembered = Remembered {
                        extended: w.extended,
                        episodes: w.episodes.len(),
                        first_episode_end: w.episodes[0].points.last().unwrap().timestamp,
                    };
                    (*key, remembered)
                })
                .collect();
            let now = FIRST + (k as u64 + 1) * 60_000;
            let start = now.saturating_sub(lookback_ms);
            full.hsl_windows.clear();
            let a = cached.update_hsl(k);
            let b = full.update_hsl(k);
            match (&a, &b) {
                (Ok(()), Ok(())) => {}
                (Err(x), Err(y)) => {
                    assert_eq!(x, y, "seed={seed} k={k}");
                    cov.errors += 1;
                    break;
                }
                _ => panic!("seed={seed} k={k}: cached={a:?} full={b:?}"),
            }
            cov.bars += 1;
            let (sig_cached, sig_full) = (signature(&cached), signature(&full));
            assert!(
                sig_cached == sig_full,
                "seed={seed} k={k}\ncached:\n{sig_cached}\nfull:\n{sig_full}"
            );
            for (key, previous) in &before {
                let (Some(side), Some(coin)) = *key else {
                    unreachable!()
                };
                let after = cached.hsl_windows.get(key);
                let hit = after.is_some_and(|w| w.extended > previous.extended);
                let name = &s.params.coins[coin];
                let selected =
                    |f: &Fill| f.coin == *name && f.order_type.is_long() == (side == LONG);
                let expired = cached.fills.iter().any(|f| {
                    selected(f) && f.timestamp_ms + 60_000 >= start && f.timestamp_ms < start
                });
                let traded = cached
                    .fills
                    .iter()
                    .rev()
                    .take_while(|f| f.index == k)
                    .any(selected);
                let episode_expired =
                    previous.episodes > 1 && (previous.first_episode_end as u64) < start;
                if hit {
                    cov.hits += 1;
                    case_hits += 1;
                    if position(&cached, side, coin).size == 0.0 {
                        cov.flat_hits += 1;
                    }
                    if previous.episodes > 1 {
                        cov.multi_episode_hits += 1;
                    }
                    assert!(
                        !expired && !traded && !episode_expired,
                        "seed={seed} k={k} {key:?}: extended a changed window"
                    );
                } else if traded {
                    cov.full_after_hit_position_change += 1;
                    // A left-truncated reconstruction restarts from zero
                    // inventory, so a reduction inside the window can zero it and
                    // split the trace although the real position never went flat.
                    if after.is_some_and(|w| w.episodes.len() > 1)
                        && position(&cached, side, coin).size != 0.0
                    {
                        cov.boundary_after_fill += 1;
                    }
                } else if expired {
                    cov.full_after_hit_fill_expiry += 1;
                } else if episode_expired {
                    cov.full_after_episode_expiry += 1;
                }
            }
            for scope in &cached.hsl_scopes {
                if let Some(d) = &scope.result.decision {
                    match d.action {
                        Action::Panic => cov.panic_decisions += 1,
                        Action::Halted => cov.halted_decisions += 1,
                        Action::Normal => {}
                    }
                }
                cov.restart_events += scope
                    .result
                    .events
                    .iter()
                    .filter(|e| e.kind == "restart")
                    .count();
            }
        }
        if case_hits > 0 {
            cov.cases_with_hits += 1;
        }
        assert_eq!(
            cached.hsl_report_value().unwrap(),
            full.hsl_report_value().unwrap(),
            "seed={seed}: report"
        );
    }

    /// Seeds are fixed (`1000..1000 + cases`), so the case set does not depend
    /// on the thread count. `HSL_WINDOW_CASES=1000` runs the longer local sweep;
    /// every coverage assertion below also holds at the default 200.
    #[test]
    fn window_extension_matches_full_reconstruction_on_random_tapes() {
        let cases: u64 = std::env::var("HSL_WINDOW_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let total = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads as u64)
                .map(|t| {
                    scope.spawn(move || {
                        let mut cov = Coverage::default();
                        let mut seed = 1_000 + t;
                        while seed < 1_000 + cases {
                            run_case(seed, &mut cov);
                            seed += threads as u64;
                        }
                        cov
                    })
                })
                .collect();
            let mut total = Coverage::default();
            for worker in workers {
                total.add(&worker.join().unwrap());
            }
            total
        });
        eprintln!("hsl window coverage: {total:?}");
        assert_eq!(total.cases as u64, cases);
        assert!(total.cases_with_hits * 2 >= total.cases, "{total:?}");
        assert!(total.flat_hits > 0, "{total:?}");
        assert!(total.multi_episode_hits > 0, "{total:?}");
        assert!(total.full_after_hit_position_change > 0, "{total:?}");
        assert!(total.full_after_hit_fill_expiry > 0, "{total:?}");
        assert!(total.boundary_after_fill > 0, "{total:?}");
        assert!(total.full_after_episode_expiry > 0, "{total:?}");
        assert!(total.panic_decisions > 0, "{total:?}");
        assert!(total.halted_decisions > 0, "{total:?}");
        assert!(total.restart_events > 0, "{total:?}");
    }
}
