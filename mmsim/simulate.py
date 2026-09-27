"""Monte Carlo market-making simulator on the Avellaneda-Stoikov grid.

Each step, for every path at once:
  1. the agent posts a bid and an ask;
  2. a market sell hits the bid with probability lambda(delta_b) dt, a market
     buy lifts the ask with probability lambda(delta_a) dt (both can happen);
  3. optionally, a filled order is 'toxic' and the mid jumps against the agent;
  4. the mid-price moves up or down by sigma * sqrt(dt).
At T the inventory is marked to the mid: P&L = cash + q * S_T.
"""

from dataclasses import dataclass

import numpy as np

from .model import average_spread, fill_intensity, inventory_quotes

STRATEGIES = ("inventory", "symmetric", "capped")


@dataclass(frozen=True)
class Params:
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    gamma: float = 0.1
    k: float = 1.5
    A: float = 140.0
    q_max: int = 5  # only used by the 'capped' strategy
    toxic_prob: float = 0.0  # share of fills followed by an adverse jump
    toxic_jump: float = 0.0  # size of that jump, in price units
    extra_spread: float = 0.0  # added to the total spread, half on each side

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))


@dataclass
class Result:
    pnl: np.ndarray
    final_q: np.ndarray
    spread_pnl: np.ndarray
    inventory_pnl: np.ndarray
    n_trades: np.ndarray
    avg_spread: float
    q_path: np.ndarray | None = None
    s_path: np.ndarray | None = None
    bid_path: np.ndarray | None = None
    ask_path: np.ndarray | None = None

    def summary(self) -> dict:
        return {
            "avg_spread": self.avg_spread,
            "pnl_mean": float(self.pnl.mean()),
            "pnl_std": float(self.pnl.std(ddof=1)),
            "final_q_mean": float(self.final_q.mean()),
            "final_q_std": float(self.final_q.std(ddof=1)),
            "spread_pnl_mean": float(self.spread_pnl.mean()),
            "inventory_pnl_mean": float(self.inventory_pnl.mean()),
            "trades_mean": float(self.n_trades.mean()),
        }


def simulate(strategy: str, p: Params, n_paths: int = 1000, seed: int | None = 0, keep_path: bool = False) -> Result:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    rng = np.random.default_rng(seed)
    n = p.n_steps
    sym_spread = average_spread(p.T, p.dt, p.gamma, p.sigma, p.k)

    s = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    spread_pnl = np.zeros(n_paths)
    n_trades = np.zeros(n_paths, dtype=int)
    spreads = np.empty(n)
    paths = {name: np.empty(n + 1) for name in ("q", "s", "bid", "ask")} if keep_path else None

    for i in range(n):
        tau = p.T - i * p.dt
        if strategy == "symmetric":
            bid, ask = s - sym_spread / 2.0, s + sym_spread / 2.0
        else:
            bid, ask = inventory_quotes(s, q, tau, p.gamma, p.sigma, p.k)
        bid, ask = bid - p.extra_spread / 2.0, ask + p.extra_spread / 2.0
        spreads[i] = float(np.mean(ask - bid))
        if keep_path:
            paths["q"][i], paths["s"][i], paths["bid"][i], paths["ask"][i] = q[0], s[0], bid[0], ask[0]

        # Distances are floored at zero: a quote through the mid is taken for sure.
        delta_b = np.maximum(s - bid, 0.0)
        delta_a = np.maximum(ask - s, 0.0)
        p_bid = np.minimum(fill_intensity(delta_b, p.A, p.k) * p.dt, 1.0)
        p_ask = np.minimum(fill_intensity(delta_a, p.A, p.k) * p.dt, 1.0)
        buy = rng.random(n_paths) < p_bid  # agent buys at the bid
        sell = rng.random(n_paths) < p_ask  # agent sells at the ask
        if strategy == "capped":
            buy &= q < p.q_max
            sell &= q > -p.q_max

        cash += np.where(sell, ask, 0.0) - np.where(buy, bid, 0.0)
        q += sell * -1.0 + buy * 1.0
        spread_pnl += np.where(sell, ask - s, 0.0) + np.where(buy, s - bid, 0.0)
        n_trades += buy.astype(int) + sell.astype(int)

        if p.toxic_prob > 0.0:
            # Informed flow: after the agent sells, price rises; after it buys, price falls.
            tox_sell = sell & (rng.random(n_paths) < p.toxic_prob)
            tox_buy = buy & (rng.random(n_paths) < p.toxic_prob)
            s = s + p.toxic_jump * (tox_sell.astype(float) - tox_buy.astype(float))

        s = s + p.sigma * np.sqrt(p.dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)

    if keep_path:
        paths["q"][n], paths["s"][n] = q[0], s[0]
        paths["bid"][n], paths["ask"][n] = paths["bid"][n - 1], paths["ask"][n - 1]

    pnl = cash + q * s
    return Result(
        pnl=pnl,
        final_q=q,
        spread_pnl=spread_pnl,
        inventory_pnl=pnl - spread_pnl,
        n_trades=n_trades,
        avg_spread=float(np.mean(spreads)),
        q_path=paths["q"] if keep_path else None,
        s_path=paths["s"] if keep_path else None,
        bid_path=paths["bid"] if keep_path else None,
        ask_path=paths["ask"] if keep_path else None,
    )
