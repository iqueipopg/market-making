# market-making

A Monte Carlo market-making simulator built on Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*, Quantitative Finance 8(3), 217-224.

It does three things:

1. **Replicates the paper.** Tables 1-3 (three risk-aversion levels, inventory strategy against a symmetric benchmark, 1000 paths each) come out within Monte Carlo noise.
2. **Splits the P&L in two.** Every trade earns its distance to the mid-price (*spread capture*); everything else is what the inventory made or lost while it was held (*inventory P&L*). The two add up to the total exactly.
3. **Adds adverse selection.** A share of fills is informed: after the agent sells, the price jumps up; after it buys, it drops. This is the risk the original model leaves out.

## The model in four lines

- The mid-price moves up or down by `sigma * sqrt(dt)` each step.
- Market orders hit a quote `delta` away from the mid with intensity `A * exp(-k * delta)`: quote wide and you rarely trade, quote tight and you trade often for little.
- The agent quotes around a **reservation price** `r = s - q * gamma * sigma^2 * (T - t)` instead of the mid. Long inventory pushes both quotes down, so it sells more and buys less.
- The total spread is `gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)`: an inventory-risk term that shrinks to zero at `T`, plus a mark-up set by how fast order flow decays.

Parameters are the paper's: `s = 100, T = 1, sigma = 2, dt = 0.005, k = 1.5, A = 140`.

## Results

## 1. Replication of Avellaneda & Stoikov (2008), Tables 1-3

| gamma | strategy | spread (paper / ours) | profit | std(profit) | final q | std(final q) |
|---|---|---|---|---|---|---|
| 0.1 | inventory | 1.49 / 1.49 | 65.0 / 65.0 | 6.6 / 6.6 | 0.08 / 0.08 | 2.9 / 2.9 |
| 0.1 | symmetric | 1.49 / 1.49 | 68.4 / 67.9 | 12.7 / 13.5 | 0.26 / -0.28 | 8.4 / 8.3 |
| 0.01 | inventory | 1.35 / 1.35 | 68.6 / 68.2 | 8.7 / 8.9 | 0.12 / 0.05 | 5.1 / 5.1 |
| 0.01 | symmetric | 1.35 / 1.35 | 68.8 / 68.4 | 12.8 / 13.9 | 0.09 / -0.24 | 8.7 / 8.7 |
| 1.0 | inventory | 3.02 / 3.03 | 31.4 / 30.8 | 5.0 / 4.9 | 0.02 / 0.04 | 1.7 / 1.6 |
| 1.0 | symmetric | 3.02 / 3.03 | 44.0 / 43.1 | 11.0 / 10.9 | 0.00 / 0.01 | 5.1 / 5.1 |

## 2. Where the P&L comes from (gamma = 0.1, 30% toxic fills, jump = 1.0)

| strategy | P&L | std | spread capture | inventory P&L | trades | std(final q) |
|---|---|---|---|---|---|---|
| symmetric | 68.1 | 13.3 | 68.0 | 0.1 | 91 | 8.1 |
| symmetric (toxic) | 47.4 | 32.5 | 68.0 | -20.7 | 91 | 8.6 |
| inventory | 64.7 | 6.6 | 64.7 | 0.0 | 97 | 3.0 |
| inventory (toxic) | 52.5 | 9.1 | 64.6 | -12.1 | 97 | 2.8 |
| capped | 64.5 | 6.6 | 64.5 | 0.0 | 96 | 2.7 |
| capped (toxic) | 52.6 | 8.9 | 64.5 | -11.8 | 96 | 2.5 |

## 3. Does widening the spread pay? (inventory strategy, 30% toxic fills)

| jump size | +0.0 | +0.5 | +1.0 | +2.0 | +3.0 |
|---|---|---|---|---|---|
| 0.0 | **64.7** | 61.0 | 53.4 | 36.1 | 22.0 |
| 1.0 | **52.5** | 51.6 | 46.3 | 32.4 | 20.1 |
| 2.0 | 40.2 | **42.2** | 39.1 | 28.5 | 18.1 |
| 3.0 | 28.0 | **32.8** | 32.0 | 24.7 | 16.1 |

### Reading the tables

- **Table 1.** The inventory strategy gives up a little mean P&L (65.0 against 67.9 at gamma = 0.1) to halve its volatility (6.6 against 13.5) and cut the dispersion of final inventory by a factor of three. At gamma = 1 the trade-off is steeper: 30.8 against 43.1, with less than half the risk. The one visible gap with the paper is the average spread at gamma = 1 (3.03 against 3.02): averaging over the discrete grid adds `gamma * sigma^2 * dt / 2 = 0.01`.
- **Table 2.** Without informed flow, both strategies make essentially all their money from the spread; inventory P&L averages zero because the mid is a martingale. With 30% toxic fills the spread income does not change, but inventory P&L turns negative: -20.7 for the symmetric quoter, with its P&L volatility going from 13 to 32. Skewing quotes by inventory limits the damage to -12.1 at a volatility of 9.
- **Table 3.** Widening the spread only pays once the flow is toxic enough. With clean flow every extra tick of width costs money. With 30% toxic fills and a jump of 2 or 3, adding 0.5 to the spread is the best of the widths tested: it trades a third less but loses much less to informed traders.

## Limitations

- Mid-price is arithmetic Brownian motion with constant volatility; no jumps other than the toxic ones.
- Fill intensity is the paper's exponential form, the same on both sides and constant in time. No queue position, no tick size, no latency.
- One asset, one lot per fill, no fees or rebates.
- Adverse selection is an added assumption, not calibrated to data. The jump sizes and the 30% share are illustrative.
- The inventory strategy is the paper's asymptotic approximation, not the exact solution of the HJB equation.

## Usage

```bash
pip install -r requirements.txt
pytest -q               # 34 tests
python experiments.py   # prints every table above
```

```python
from mmsim import Params, simulate

r = simulate("inventory", Params(gamma=0.1, toxic_prob=0.3, toxic_jump=1.0), n_paths=2000, seed=0)
print(r.summary())
```

Strategies: `inventory` (Avellaneda-Stoikov quotes), `symmetric` (same average spread, centred on the mid), `capped` (inventory quotes with a hard position limit `q_max`).

## License

MIT
