import numpy as np
import pytest

from mmsim import Params, simulate

# Avellaneda & Stoikov (2008), Tables 1-3: (profit mean, profit std, final q std), 1000 paths.
PAPER = {
    (0.1, "inventory"): (65.0, 6.6, 2.9),
    (0.1, "symmetric"): (68.4, 12.7, 8.4),
    (0.01, "inventory"): (68.6, 8.7, 5.1),
    (0.01, "symmetric"): (68.8, 12.8, 8.7),
    (1.0, "inventory"): (31.4, 5.0, 1.7),
    (1.0, "symmetric"): (44.0, 11.0, 5.1),
}


@pytest.mark.parametrize("gamma, strategy", list(PAPER))
def test_replicates_paper_tables(gamma, strategy):
    mean, std, q_std = PAPER[(gamma, strategy)]
    r = simulate(strategy, Params(gamma=gamma), n_paths=4000, seed=7)
    se = std / np.sqrt(4000)
    # Mean within 4 standard errors of the paper's 1000-path estimate (which has its own noise),
    # dispersion within 12%.
    assert abs(r.pnl.mean() - mean) < 4 * se + 1.0
    assert r.pnl.std(ddof=1) == pytest.approx(std, rel=0.12)
    assert r.final_q.std(ddof=1) == pytest.approx(q_std, rel=0.12)


@pytest.mark.parametrize("gamma", [0.1, 0.01, 1.0])
def test_inventory_strategy_has_lower_risk_than_symmetric(gamma):
    inv = simulate("inventory", Params(gamma=gamma), n_paths=2000, seed=3)
    sym = simulate("symmetric", Params(gamma=gamma), n_paths=2000, seed=3)
    assert inv.pnl.std() < sym.pnl.std()
    assert inv.final_q.std() < sym.final_q.std()


def test_same_average_spread_for_both_strategies():
    inv = simulate("inventory", Params(), n_paths=200, seed=0)
    sym = simulate("symmetric", Params(), n_paths=200, seed=0)
    assert inv.avg_spread == pytest.approx(sym.avg_spread, abs=0.01)


def test_seed_makes_runs_reproducible():
    a = simulate("inventory", Params(), n_paths=100, seed=42)
    b = simulate("inventory", Params(), n_paths=100, seed=42)
    np.testing.assert_array_equal(a.pnl, b.pnl)


def test_pnl_decomposition_adds_up():
    r = simulate("symmetric", Params(toxic_prob=0.2, toxic_jump=1.0), n_paths=300, seed=1)
    np.testing.assert_allclose(r.spread_pnl + r.inventory_pnl, r.pnl)


def test_spread_capture_is_positive_per_trade():
    r = simulate("symmetric", Params(), n_paths=300, seed=1)
    assert np.all(r.spread_pnl >= 0)
    per_trade = r.spread_pnl.sum() / r.n_trades.sum()
    assert per_trade == pytest.approx(r.avg_spread / 2, rel=1e-9)


def test_no_order_flow_means_no_trades():
    r = simulate("inventory", Params(A=0.0), n_paths=100, seed=0)
    assert r.n_trades.sum() == 0
    np.testing.assert_array_equal(r.pnl, 0.0)


def test_no_volatility_and_no_inventory_risk_gives_pure_spread_income():
    r = simulate("symmetric", Params(sigma=0.0), n_paths=200, seed=0)
    np.testing.assert_allclose(r.inventory_pnl, 0.0, atol=1e-9)


def test_cap_bounds_inventory():
    r = simulate("capped", Params(q_max=3, gamma=0.001), n_paths=500, seed=5, keep_path=False)
    assert np.all(np.abs(r.final_q) <= 3)


def test_toxic_flow_costs_money_through_inventory():
    clean = simulate("inventory", Params(), n_paths=2000, seed=9)
    toxic = simulate("inventory", Params(toxic_prob=0.3, toxic_jump=1.0), n_paths=2000, seed=9)
    assert toxic.inventory_pnl.mean() < clean.inventory_pnl.mean() - 5
    assert toxic.spread_pnl.mean() == pytest.approx(clean.spread_pnl.mean(), rel=0.02)


def test_wider_spread_pays_when_flow_is_very_toxic():
    p = dict(toxic_prob=0.3, toxic_jump=3.0)
    tight = simulate("inventory", Params(**p), n_paths=2000, seed=4)
    wide = simulate("inventory", Params(**p, extra_spread=0.5), n_paths=2000, seed=4)
    assert wide.pnl.mean() > tight.pnl.mean()


def test_wider_spread_costs_money_when_flow_is_clean():
    tight = simulate("inventory", Params(), n_paths=2000, seed=4)
    wide = simulate("inventory", Params(extra_spread=1.0), n_paths=2000, seed=4)
    assert wide.pnl.mean() < tight.pnl.mean()


def test_keep_path_records_one_path():
    r = simulate("inventory", Params(), n_paths=5, seed=0, keep_path=True)
    assert r.s_path.shape == (Params().n_steps + 1,)
    assert np.all(r.bid_path < r.ask_path)


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        simulate("martingale", Params())
