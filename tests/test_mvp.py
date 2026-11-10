"""Five MVP checks using invented prices only, never market or research results."""

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import can_trade, run_backtest, transaction_cost
from src.evaluate import calculate_rank_ic
from src.factors import build_signals


CODES = tuple(f"sh.{600000 + index:06d}" for index in range(10))
FACTORS = ["momentum", "reversal", "low_volatility", "turnover"]


def _price_panel(calendar):
    """Construct ten distinct complete stock paths on an invented session grid."""
    time = np.arange(len(calendar), dtype=float)
    pieces = []
    for index, code in enumerate(CODES):
        close = 15 + index + 0.012 * (index + 1) * time + 0.2 * np.sin(time / (7 + index))
        pieces.append(pd.DataFrame({
            "date": calendar, "code": code,
            "open": close, "high": close + 0.1, "low": close - 0.1,
            "close": close, "preclose": np.r_[close[0], close[:-1]],
            "adj_close": close, "volume": 1_000_000,
            "turn": 1 + 0.1 * index + 0.3 * np.sin(time / (5 + index)),
            "isST": 0, "tradestatus": 1,
        }))
    return pd.concat(pieces, ignore_index=True)


def _monthly_signals(dates, reverse_second=False):
    """Create explicit month-end rankings for ten hypothetical stocks."""
    rows = []
    for position, date in enumerate(dates):
        for index, code in enumerate(CODES):
            score = 10 - index if reverse_second and position == 1 else index + 1
            rows.append({"date": pd.Timestamp(date), "code": code, "score": float(score)})
    return pd.DataFrame(rows)


def test_future_prices_do_not_change_past_signals():
    """A future perturbation or truncation cannot alter factors already known."""
    calendar = pd.bdate_range("2017-01-02", "2019-03-29")
    prices = _price_panel(calendar)
    before = prices.copy(deep=True)
    month_ends = pd.Series(calendar).groupby(calendar.to_period("M")).max()
    month_ends = month_ends.loc[month_ends.ge(pd.Timestamp("2018-01-01"))]
    constituents = pd.MultiIndex.from_product(
        [month_ends.to_numpy(), CODES], names=["signal_date", "code"]
    ).to_frame(index=False)
    cutoff = pd.Timestamp("2018-08-31")
    baseline = build_signals(prices, constituents)
    past = baseline.loc[baseline["date"].le(cutoff)].reset_index(drop=True)
    assert len(past) == 8 * len(CODES)
    assert np.isfinite(past[FACTORS + ["score"]].to_numpy()).all()
    pd.testing.assert_frame_equal(prices, before)

    changed = prices.copy(deep=True)
    future = changed["date"].gt(cutoff)
    price_columns = ["open", "high", "low", "close", "preclose", "adj_close"]
    # Vary by both stock and date so a common rescaling cannot hide leakage.
    stock_scale = changed.loc[future, "code"].map(dict(zip(CODES, range(1, 11))))
    time_scale = 1 + (changed.loc[future, "date"] - cutoff).dt.days / 100
    changed.loc[future, price_columns] = changed.loc[future, price_columns].mul(
        2 + stock_scale * time_scale, axis=0
    )
    changed.loc[future, "turn"] += stock_scale ** 2 * time_scale
    perturbed = build_signals(changed, constituents)
    pd.testing.assert_frame_equal(
        past, perturbed.loc[perturbed["date"].le(cutoff)].reset_index(drop=True),
        check_exact=True,
    )
    truncated = build_signals(
        prices.loc[prices["date"].le(cutoff)].copy(),
        constituents.loc[constituents["signal_date"].le(cutoff)].copy(),
    )
    pd.testing.assert_frame_equal(past, truncated.reset_index(drop=True), check_exact=True)

    history = prices.loc[prices["code"].eq(CODES[0]) & prices["date"].le(cutoff)]
    close = history["adj_close"].reset_index(drop=True)
    actual = past.loc[past["date"].eq(cutoff) & past["code"].eq(CODES[0])].iloc[0]
    assert actual["momentum"] == pytest.approx(close.iloc[-22] / close.iloc[-253] - 1)
    assert actual["reversal"] == pytest.approx(-(close.iloc[-1] / close.iloc[-22] - 1))
    assert actual["low_volatility"] == pytest.approx(-close.pct_change(fill_method=None).iloc[-60:].std(ddof=1))
    assert actual["turnover"] == pytest.approx(-history["turn"].iloc[-20:].mean())
    last = past.loc[past["date"].eq(cutoff)].set_index("code")
    expected_score = last[FACTORS].rank(method="average", pct=True).mean(axis=1)
    np.testing.assert_allclose(last["score"], expected_score)

    # An old missing row must not split ties in otherwise identical windows.
    tie_calendar = pd.bdate_range(end="2020-12-31", periods=253)
    time = np.arange(len(tie_calendar), dtype=float)
    tie_prices = pd.concat([
        pd.DataFrame({
            "date": tie_calendar, "code": code,
            "adj_close": 30 + 0.02 * scale * time + 0.5 * np.sin(time / (5 + scale)),
            "turn": 2 + 0.3 * scale + 0.1 * np.cos(time / 7),
            "isST": 0, "tradestatus": 1,
        })
        for code, scale in zip(CODES[:3], [1, 1, 2])
    ], ignore_index=True)
    tie_members = pd.DataFrame({"signal_date": tie_calendar[-1], "code": CODES[:3]})
    complete_ties = build_signals(tie_prices, tie_members).set_index("code")
    old_missing_row = tie_prices["code"].eq(CODES[0]) & tie_prices["date"].eq(tie_calendar[50])
    missing_ties = build_signals(tie_prices.loc[~old_missing_row].copy(), tie_members).set_index("code")
    pd.testing.assert_frame_equal(complete_ties, missing_ties, check_exact=True)
    assert missing_ties.loc[CODES[0], "low_volatility"] == missing_ties.loc[CODES[1], "low_volatility"]
    assert missing_ties.loc[CODES[0], "score"] == missing_ties.loc[CODES[1], "score"]
    np.testing.assert_allclose(missing_ties.loc[list(CODES[:2]), "score"], 2 / 3, rtol=0, atol=1e-15)


def test_signals_align_to_next_session_open_and_complete_monthly_windows():
    """Use the supplied calendar and exact future opens, with no terminal filling."""
    # This is an invented session grid, not an exchange holiday calendar.
    calendar = pd.bdate_range("2019-01-02", "2019-03-29").difference(
        pd.DatetimeIndex(["2019-02-01"])
    )
    prices = _price_panel(calendar)
    level = pd.Series(10.0, index=calendar)
    level.loc[pd.Timestamp("2019-01-31")] = 9.9
    level.loc[level.index >= pd.Timestamp("2019-02-05")] = 10.5
    level.loc[level.index >= pd.Timestamp("2019-03-01")] = 11.0
    for field in ["open", "high", "low", "close", "adj_close"]:
        prices[field] = prices["date"].map(level)
    closing_level = level.copy()
    closing_level.loc[pd.Timestamp("2019-02-04")] = 10.2
    closing_level.loc[pd.Timestamp("2019-03-01")] = 11.3
    for field in ["close", "adj_close"]:
        prices[field] = prices["date"].map(closing_level)
    prices["high"] = prices[["open", "close"]].max(axis=1)
    prices["low"] = prices[["open", "close"]].min(axis=1)
    prices["preclose"] = prices["date"].map(closing_level.shift().fillna(closing_level.iloc[0]))
    signals = _monthly_signals(["2019-01-31", "2019-02-28", "2019-03-29"], reverse_second=True)
    nav, trades, forward = run_backtest(prices, signals, calendar, "2019-01-02", "2019-03-29", 0)
    filled = trades.loc[trades["reason"].eq("rebalance")]
    assert not filled.empty
    assert set(pd.to_datetime(filled["date"])) == {
        pd.Timestamp("2019-02-04"), pd.Timestamp("2019-03-01")
    }
    assert not pd.to_datetime(filled["date"]).isin(pd.to_datetime(signals["date"])).any()
    assert not pd.to_datetime(filled["date"]).eq(pd.Timestamp("2019-02-01")).any()
    np.testing.assert_allclose(nav.loc[:"2019-01-31"], 1.0)
    np.testing.assert_allclose(nav.loc["2019-02-04"], 10.2 / 10.0)
    np.testing.assert_allclose(nav.loc["2019-03-01"], 11.3 / 10.0)
    assert len(forward) == len(signals)
    assert not forward.duplicated(["date", "code"]).any()
    january = forward.loc[pd.to_datetime(forward["date"]).eq(pd.Timestamp("2019-01-31"))]
    assert set(january["code"]) == set(CODES)
    np.testing.assert_allclose(january["forward_return"], 11.0 / 10.0 - 1)
    unfinished = forward.loc[pd.to_datetime(forward["date"]).ge(pd.Timestamp("2019-02-28"))]
    assert set(pd.to_datetime(unfinished["date"])) == {pd.Timestamp("2019-02-28"), pd.Timestamp("2019-03-29")}
    assert unfinished["forward_return"].isna().all()

    for missing_date in ["2019-02-04", "2019-03-01"]:
        missing_open = prices.copy(deep=True)
        missing_open.loc[
            missing_open["code"].eq(CODES[0]) & missing_open["date"].eq(pd.Timestamp(missing_date)), "open"
        ] = np.nan
        _, _, missing_forward = run_backtest(missing_open, signals, calendar, "2019-01-02", "2019-03-29", 0)
        january_missing = missing_forward.loc[
            pd.to_datetime(missing_forward["date"]).eq(pd.Timestamp("2019-01-31"))
        ].set_index("code")
        assert len(january_missing) == len(CODES)
        assert pd.isna(january_missing.loc[CODES[0], "forward_return"])
        np.testing.assert_allclose(january_missing.drop(index=CODES[0])["forward_return"], 11.0 / 10.0 - 1)

    with pytest.raises(ValueError, match="month-end"):
        run_backtest(
            prices, _monthly_signals(["2019-01-15"]), calendar,
            "2019-01-02", "2019-03-29", 0,
        )

    calendar_without_february = calendar[calendar.to_period("M") != pd.Period("2019-02", freq="M")]
    prices_without_february = prices.loc[prices["date"].isin(calendar_without_february)].copy()
    with pytest.raises(ValueError, match="missing the month immediately after"):
        run_backtest(
            prices_without_february, _monthly_signals(["2019-01-31"]), calendar_without_february,
            "2019-01-02", "2019-03-29", 0,
        )


def test_rank_ic_matches_ten_stock_hand_calculation():
    """Check the ten-stock Spearman formula, average ties, and unavailable pairs."""
    ranks = np.arange(1.0, 11.0)
    permutation = np.array([1, 3, 2, 4, 5, 6, 7, 8, 10, 9], dtype=float)
    signals = pd.DataFrame({"date": pd.Timestamp("2020-01-31"), "code": CODES})
    for factor in FACTORS + ["score"]:
        # Nonlinear values distinguish rank correlation from raw Pearson IC.
        signals[factor] = ranks ** 2
    forward = signals[["date", "code"]].copy()
    forward["forward_return"] = permutation ** 3 / 1000
    result = calculate_rank_ic(signals.sample(frac=1, random_state=7), forward.sample(frac=1, random_state=3))
    assert set(result["factor"]) == set(FACTORS + ["score"])
    assert result["n"].eq(10).all()
    # Two adjacent swaps give sum(d_i**2)=4, hence rho=97/99.
    expected = 1 - 6 * 4 / (10 * 99)
    assert len(result) == len(FACTORS) + 1
    assert pd.to_datetime(result["date"]).eq(pd.Timestamp("2020-01-31")).all()
    np.testing.assert_allclose(result["rank_ic"], expected)

    tied = signals.copy(deep=True)
    tied["momentum"] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    tied["turnover"] = 1.0
    tied_result = calculate_rank_ic(tied, forward).set_index("factor")
    centered_x = np.array([1.5, 1.5, 3, 4, 5, 6, 7, 8, 9, 10]) - 5.5
    centered_y = permutation - 5.5
    tied_expected = float(np.dot(centered_x, centered_y) / math.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)))
    assert tied_result.loc["momentum", "rank_ic"] == pytest.approx(tied_expected)
    assert pd.isna(tied_result.loc["turnover", "rank_ic"])
    forward.loc[0, "forward_return"] = np.nan
    paired = calculate_rank_ic(tied, forward).set_index("factor")
    assert paired["n"].eq(9).all()
    # Removing the first pair leaves nine distinct ranks and the same two swaps.
    np.testing.assert_allclose(paired.loc[["momentum", "reversal", "low_volatility", "score"], "rank_ic"], 29 / 30)
    assert pd.isna(paired.loc["turnover", "rank_ic"])


def test_trade_gate_handles_limits_suspension_and_cent_rounding():
    """Gate raw opening prices by board/date, state, direction, and cent rounding."""
    cases = [
        ("sh.600000", 0, "2020-08-24", 11.0, 9.0),
        ("sh.600000", 1, "2020-08-24", 10.5, 9.5),
        ("sz.000001", 0, "2020-08-24", 11.0, 9.0),
        ("sz.000001", 1, "2020-08-24", 10.5, 9.5),
        ("sh.688001", 0, "2020-08-24", 12.0, 8.0),
        ("sh.688001", 1, "2020-08-24", 12.0, 8.0),
        ("sz.300001", 0, "2020-08-21", 11.0, 9.0),
        ("sz.300001", 1, "2020-08-21", 10.5, 9.5),
        ("sz.300001", 0, "2020-08-24", 12.0, 8.0),
        ("sz.300001", 1, "2020-08-24", 12.0, 8.0),
        ("sz.301001", 0, "2021-08-24", 12.0, 8.0),
        ("sz.301001", 1, "2021-08-24", 12.0, 8.0),
        ("sz.302132", 0, "2025-12-31", 12.0, 8.0),
    ]
    for code, is_st, date, upper, lower in cases:
        row = {"code": code, "isST": is_st, "tradestatus": 1, "volume": 1_000_000, "preclose": 10.0, "open": upper}
        assert not can_trade(row, "buy", date), (code, is_st, date)
        assert can_trade(row, "sell", date), (code, is_st, date)
        assert not can_trade({**row, "open": upper + 0.01}, "buy", date), (code, is_st, date)
        row["open"] = upper - 0.01
        assert can_trade(row, "buy", date), (code, is_st, date)
        row["open"] = lower
        assert not can_trade(row, "sell", date), (code, is_st, date)
        assert can_trade(row, "buy", date), (code, is_st, date)
        assert not can_trade({**row, "open": lower - 0.01}, "sell", date), (code, is_st, date)
        row["open"] = lower + 0.01
        assert can_trade(row, "sell", date), (code, is_st, date)

    row = {"code": "sh.600000", "isST": 0, "tradestatus": 1, "volume": 1_000_000, "preclose": 10.05, "open": 11.05}
    assert can_trade(row, "buy", "2020-08-24")
    assert not can_trade({**row, "open": 11.06}, "buy", "2020-08-24")
    assert not can_trade({**row, "open": 9.05}, "sell", "2020-08-24")
    assert can_trade({**row, "open": 9.06}, "sell", "2020-08-24")
    for side in ["buy", "sell"]:
        assert not can_trade({**row, "open": 10.05, "tradestatus": 0}, side, "2020-08-24")
        assert not can_trade({**row, "open": 10.05, "tradestatus": pd.NA}, side, "2020-08-24")
        assert not can_trade({**row, "open": 10.05, "isST": pd.NA}, side, "2020-08-24")
        assert not can_trade({**row, "open": np.nan}, side, "2020-08-24")
        assert not can_trade({**row, "volume": 0}, side, "2020-08-24")
        assert not can_trade({**row, "preclose": np.nan}, side, "2020-08-24")

    calendar = pd.bdate_range("2019-01-02", "2019-03-05")
    prices = _price_panel(calendar)
    for field in ["open", "high", "low", "close", "preclose", "adj_close"]:
        prices[field] = 10.0
    blocked_buy = prices["date"].eq(pd.Timestamp("2019-02-01")) & prices["code"].eq(CODES[0])
    blocked_sell = prices["date"].eq(pd.Timestamp("2019-03-01")) & prices["code"].eq(CODES[1])
    prices.loc[blocked_buy, ["open", "high"]] = 11.0
    prices.loc[blocked_sell, ["open", "low"]] = 9.0
    signals = _monthly_signals(["2019-01-31", "2019-02-28"], reverse_second=True)
    nav, trades, _ = run_backtest(prices, signals, calendar, "2019-01-02", "2019-03-05", 0)
    held_purchase = trades.loc[
        trades["portfolio"].eq("Q1") & trades["code"].eq(CODES[1])
        & pd.to_datetime(trades["date"]).eq(pd.Timestamp("2019-02-01"))
        & trades["side"].eq("buy") & trades["reason"].eq("rebalance")
    ]
    assert len(held_purchase) == 1
    assert held_purchase.iloc[0]["notional"] == pytest.approx(0.5)
    for date, code, side, reason in [
        ("2019-02-01", CODES[0], "buy", "limit_up"),
        ("2019-03-01", CODES[1], "sell", "limit_down"),
    ]:
        orders = trades.loc[
            trades["portfolio"].eq("Q1") & trades["code"].eq(code)
            & pd.to_datetime(trades["date"]).eq(pd.Timestamp(date)) & trades["side"].eq(side)
        ]
        assert len(orders) == 1
        assert orders.iloc[0]["reason"] == reason
        assert orders.iloc[0]["notional"] == 0
        assert orders.iloc[0]["cost"] == 0
    # The blocked sale must leave the existing holding in the account.
    assert nav.loc["2019-03-01", "Q1"] == pytest.approx(1.0)


def test_transaction_costs_and_cash_constraint():
    """Charge both trade directions consistently without borrowing cash for fees."""
    assert transaction_cost(10_000, 10) == pytest.approx(10.0)
    assert transaction_cost(0, 10) == 0
    assert transaction_cost(10_000, 0) == 0
    for notional, cost_bps in [(-1, 10), (1, -1), (np.nan, 10), (1, np.inf)]:
        with pytest.raises(ValueError):
            transaction_cost(notional, cost_bps)

    calendar = pd.bdate_range("2019-01-02", "2019-03-05")
    prices = _price_panel(calendar)
    for field in ["open", "high", "low", "close", "preclose", "adj_close"]:
        prices[field] = 10.0
    signals = _monthly_signals(["2019-01-31", "2019-02-28"], reverse_second=True)
    nav, trades, _ = run_backtest(prices, signals, calendar, "2019-01-02", "2019-03-05", 10)
    filled = trades.loc[trades["reason"].eq("rebalance")]
    assert set(filled["side"]) == {"buy", "sell"}
    assert filled["notional"].gt(0).all()
    assert filled["cost"].ge(0).all()
    np.testing.assert_allclose(filled["cost"], filled["notional"] * 0.001)
    first = filled.loc[pd.to_datetime(filled["date"]).eq(pd.to_datetime(filled["date"]).min())]
    assert first["side"].eq("buy").all()
    spending = first.assign(spending=first["notional"] + first["cost"]).groupby("portfolio")["spending"].sum()
    assert set(spending.index) == {"Q1", "Q2", "Q3", "Q4", "Q5", "benchmark"}
    np.testing.assert_allclose(spending, 1.0, rtol=0, atol=1e-12)
    assert set(nav.attrs["min_cash"]).issuperset({"Q1", "Q2", "Q3", "Q4", "Q5"})
    assert min(nav.attrs["min_cash"].values()) >= -1e-12
    assert min(nav.attrs["ending_cash"].values()) >= -1e-12
    assert nav[["Q1", "Q2", "Q3", "Q4", "Q5"]].iloc[-1].between(0, 1, inclusive="neither").all()
    for portfolio, orders in filled.groupby("portfolio", sort=False):
        # Replay every fill in execution order, independently of reported cash.
        cash_changes = np.where(
            orders["side"].eq("sell"), orders["notional"] - orders["cost"],
            -orders["notional"] - orders["cost"],
        )
        cash_after_each_fill = 1.0 + np.cumsum(cash_changes)
        assert np.isfinite(cash_after_each_fill).all()
        assert (cash_after_each_fill >= -1e-12).all(), portfolio
        assert nav.attrs["ending_cash"][portfolio] == pytest.approx(cash_after_each_fill[-1], abs=1e-12)
        assert nav[portfolio].iloc[-1] == pytest.approx(1.0 - orders["cost"].sum())

    rate = 0.001
    np.testing.assert_allclose(nav.loc["2019-02-01"], 1 / (1 + rate), rtol=0, atol=1e-12)
    # Full rotation pays the initial buy, the sale, and the replacement buy.
    np.testing.assert_allclose(
        nav[["Q1", "Q2", "Q4", "Q5"]].iloc[-1], (1 - rate) / (1 + rate) ** 2,
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(nav[["Q3", "benchmark"]].iloc[-1], 1 / (1 + rate), rtol=0, atol=1e-12)
