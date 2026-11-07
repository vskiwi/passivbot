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
        let (Some(side), Some(coin)) = key else { return };
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
                || e.points.iter().any(|p| p.cashflow_reference_delta.is_some())
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
        if self.fills[window.fill_range.0..fill_start].iter().any(selected)
            || self.fills[window.fill_range.1..fill_end].iter().any(selected)
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
        let historical = valuation.pnl(direction * window.tail_size, window.tail_basis, window.mark);
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
