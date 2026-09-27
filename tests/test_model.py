import numpy as np
import pytest

from mmsim import average_spread, fill_intensity, inventory_quotes, optimal_spread, reservation_price

SIGMA, K, T, DT = 2.0, 1.5, 1.0, 0.005


@pytest.mark.parametrize("gamma, paper", [(0.1, 1.49), (0.01, 1.35), (1.0, 3.02)])
def test_average_spread_matches_paper_tables(gamma, paper):
    # Tables 1-3 of Avellaneda & Stoikov (2008) report these average spreads. Averaging over the
    # discrete grid t = 0, dt, ..., T - dt adds gamma * sigma^2 * dt / 2 to the continuous average,
    # which is 0.01 at gamma = 1: the paper's 3.02 is the continuous value, ours is 3.03.
    assert average_spread(T, DT, gamma, SIGMA, K) == pytest.approx(paper, abs=0.015)


def test_reservation_price_is_mid_when_flat():
    assert reservation_price(100.0, 0, 0.7, 0.1, SIGMA) == 100.0


def test_long_inventory_lowers_reservation_price():
    assert reservation_price(100.0, 3, 0.5, 0.1, SIGMA) < 100.0
    assert reservation_price(100.0, -3, 0.5, 0.1, SIGMA) > 100.0


def test_reservation_shift_is_linear_in_inventory():
    shifts = [100.0 - reservation_price(100.0, q, 0.5, 0.1, SIGMA) for q in (1, 2, 4)]
    assert shifts[1] == pytest.approx(2 * shifts[0])
    assert shifts[2] == pytest.approx(4 * shifts[0])


def test_inventory_effect_vanishes_at_maturity():
    assert reservation_price(100.0, 10, 0.0, 0.1, SIGMA) == 100.0


def test_spread_at_maturity_is_pure_markup():
    gamma = 0.1
    assert optimal_spread(0.0, gamma, SIGMA, K) == pytest.approx((2 / gamma) * np.log(1 + gamma / K))


def test_spread_widens_with_time_to_maturity():
    taus = np.linspace(0.0, 1.0, 11)
    assert np.all(np.diff(optimal_spread(taus, 0.1, SIGMA, K)) > 0)


def test_markup_tends_to_two_over_k_for_risk_neutral_agent():
    # As gamma -> 0, (2/gamma) ln(1 + gamma/k) -> 2/k.
    assert optimal_spread(0.0, 1e-8, SIGMA, K) == pytest.approx(2 / K, rel=1e-6)


def test_quotes_are_symmetric_around_reservation_price():
    bid, ask = inventory_quotes(100.0, 2, 0.4, 0.1, SIGMA, K)
    r = reservation_price(100.0, 2, 0.4, 0.1, SIGMA)
    assert (bid + ask) / 2 == pytest.approx(r)
    assert ask - bid == pytest.approx(optimal_spread(0.4, 0.1, SIGMA, K))


def test_fill_intensity_decays_exponentially():
    assert fill_intensity(0.0, 140.0, K) == 140.0
    assert fill_intensity(1.0, 140.0, K) / fill_intensity(0.0, 140.0, K) == pytest.approx(np.exp(-K))


def test_functions_are_vectorised():
    q = np.array([-2.0, 0.0, 2.0])
    bid, ask = inventory_quotes(np.full(3, 100.0), q, 0.5, 0.1, SIGMA, K)
    assert bid.shape == ask.shape == (3,)
    assert bid[0] > bid[1] > bid[2]
