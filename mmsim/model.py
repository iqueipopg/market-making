"""Avellaneda-Stoikov (2008) quotes: reservation price and optimal spread.

All functions are vectorised over numpy arrays.
"""

import numpy as np


def reservation_price(s, q, tau, gamma, sigma):
    """Indifference price r = s - q * gamma * sigma^2 * tau, with tau = T - t.

    A long inventory (q > 0) pushes r below the mid-price, so the agent
    quotes lower on both sides to sell more and buy less.
    """
    return s - q * gamma * sigma**2 * tau


def optimal_spread(tau, gamma, sigma, k):
    """Total bid-ask spread delta_a + delta_b = gamma sigma^2 tau + (2/gamma) ln(1 + gamma/k).

    The first term is inventory risk (shrinks to zero at T); the second is
    the monopoly mark-up set by how fast order flow decays with distance (k).
    """
    return gamma * sigma**2 * tau + (2.0 / gamma) * np.log1p(gamma / k)


def inventory_quotes(s, q, tau, gamma, sigma, k):
    """Bid and ask of the 'inventory' strategy: centred on r, not on s."""
    r = reservation_price(s, q, tau, gamma, sigma)
    half = optimal_spread(tau, gamma, sigma, k) / 2.0
    return r - half, r + half


def average_spread(T, dt, gamma, sigma, k):
    """Time-average of the optimal spread over the simulation grid.

    The 'symmetric' benchmark in the paper quotes this constant spread
    around the mid-price, so both strategies post the same average width.
    """
    tau = T - np.arange(0.0, T, dt)
    return float(np.mean(optimal_spread(tau, gamma, sigma, k)))


def fill_intensity(delta, A, k):
    """Poisson arrival rate of market orders hitting a quote delta away from mid."""
    return A * np.exp(-k * delta)
