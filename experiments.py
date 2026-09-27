"""Reproduce every table in the README.

python experiments.py
"""

from mmsim import Params, simulate

PAPER = {
    (0.1, "inventory"): (1.49, 65.0, 6.6, 0.08, 2.9),
    (0.1, "symmetric"): (1.49, 68.4, 12.7, 0.26, 8.4),
    (0.01, "inventory"): (1.35, 68.6, 8.7, 0.12, 5.1),
    (0.01, "symmetric"): (1.35, 68.8, 12.8, 0.09, 8.7),
    (1.0, "inventory"): (3.02, 31.4, 5.0, 0.02, 1.7),
    (1.0, "symmetric"): (3.02, 44.0, 11.0, 0.00, 5.1),
}
N, SEED = 1000, 1


def replication():
    print("## 1. Replication of Avellaneda & Stoikov (2008), Tables 1-3\n")
    print("| gamma | strategy | spread (paper / ours) | profit | std(profit) | final q | std(final q) |")
    print("|---|---|---|---|---|---|---|")
    for (gamma, strategy), (sp, m, sd, q, qsd) in PAPER.items():
        r = simulate(strategy, Params(gamma=gamma), n_paths=N, seed=SEED).summary()
        print(
            f"| {gamma} | {strategy} | {sp:.2f} / {r['avg_spread']:.2f} | {m:.1f} / {r['pnl_mean']:.1f} "
            f"| {sd:.1f} / {r['pnl_std']:.1f} | {q:.2f} / {r['final_q_mean']:.2f} "
            f"| {qsd:.1f} / {r['final_q_std']:.1f} |"
        )


def decomposition():
    print("\n## 2. Where the P&L comes from (gamma = 0.1, 30% toxic fills, jump = 1.0)\n")
    print("| strategy | P&L | std | spread capture | inventory P&L | trades | std(final q) |")
    print("|---|---|---|---|---|---|---|")
    for strategy in ("symmetric", "inventory", "capped"):
        for toxic in (0.0, 0.3):
            p = Params(toxic_prob=toxic, toxic_jump=1.0)
            r = simulate(strategy, p, n_paths=2000, seed=2).summary()
            label = f"{strategy}{' (toxic)' if toxic else ''}"
            print(
                f"| {label} | {r['pnl_mean']:.1f} | {r['pnl_std']:.1f} | {r['spread_pnl_mean']:.1f} "
                f"| {r['inventory_pnl_mean']:.1f} | {r['trades_mean']:.0f} | {r['final_q_std']:.1f} |"
            )


def widening():
    print("\n## 3. Does widening the spread pay? (inventory strategy, 30% toxic fills)\n")
    widths = (0.0, 0.5, 1.0, 2.0, 3.0)
    print("| jump size | " + " | ".join(f"+{w}" for w in widths) + " |")
    print("|---|" + "---|" * len(widths))
    for jump in (0.0, 1.0, 2.0, 3.0):
        cells = []
        for w in widths:
            p = Params(toxic_prob=0.3 if jump else 0.0, toxic_jump=jump, extra_spread=w)
            cells.append(f"{simulate('inventory', p, n_paths=2000, seed=2).pnl.mean():.1f}")
        best = max(range(len(cells)), key=lambda i: float(cells[i]))
        cells[best] = f"**{cells[best]}**"
        print(f"| {jump} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    replication()
    decomposition()
    widening()
