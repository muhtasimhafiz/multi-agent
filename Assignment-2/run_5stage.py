"""5-stage curriculum adapted from the working solution, retaining my env.

Differences from the reference notebook:
- Same spec layout for X/Y/U/V/Lake (no axis swap)
- Full-grid collision detection (A-B in ANY cell counts, not just the lake)
- p_lake_flip = 0.5 (matches the reference; lake state near-IID)
- 5 stages: A solo, B solo, joint medium, joint dense, low-eps fine-tune
- Random tie-break in greedy (no WAIT preference)
- Standard Q-learning, no expected Bellman, no RNG seeding
"""

import json
import os

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B
from qlearn import QLearner


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_5stage")
os.makedirs(OUT_DIR, exist_ok=True)


def _reset_schedule(lr, alpha_start, alpha_min, alpha_decay_steps,
                    eps_start, eps_min, eps_decay_steps):
    """Reset a QLearner's alpha/eps schedule for a new stage (Q-table is kept)."""
    lr.alpha = alpha_start
    lr._alpha_start = alpha_start
    lr._alpha_min = alpha_min
    lr._alpha_decay_steps = max(1, alpha_decay_steps)
    lr.eps = eps_start
    lr._eps_start = eps_start
    lr.eps_min = eps_min
    lr._eps_decay_steps = max(1, eps_decay_steps)
    lr._steps = 0


def main():
    base_env = dict(
        p_lake_flip=0.5,           # KEY: high flip rate makes WAIT cheap
        r_step=-5,
        r_wait=-3,
        r_collision=-20,
        r_water=-20,
        r_pickup=10,
        r_deliver=50,
    )
    # Initial learners with zero init (matches reference). Optimistic init
    # not needed here; pre-training drives exploration.
    learners = {
        TYPE_A: QLearner(q_init=0.0, alpha=0.2, alpha_min=0.02, alpha_decay_steps=200_000,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=200_000),
        TYPE_B: QLearner(q_init=0.0, alpha=0.2, alpha_min=0.02, alpha_decay_steps=200_000,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=200_000),
    }

    # =======================================================================
    # STAGE 1: A solo
    # =======================================================================
    print("=" * 70)
    print("STAGE 1: A solo (no B in env)")
    print("=" * 70)
    env_s1 = {**base_env, "enabled_types": (TYPE_A,), "target_active": 1}
    _, learners, h1 = train(
        num_steps=200_000, log_window=20_000,
        env_kwargs=env_s1, learners=learners, verbose=True,
    )
    print()
    print_policy(learners[TYPE_A], "A (after stage 1)")

    # =======================================================================
    # STAGE 2: B solo
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: B solo (no A in env)")
    print("=" * 70)
    _reset_schedule(learners[TYPE_B],
                    alpha_start=0.2, alpha_min=0.02, alpha_decay_steps=160_000,
                    eps_start=1.0, eps_min=0.01, eps_decay_steps=160_000)
    env_s2 = {**base_env, "enabled_types": (TYPE_B,), "target_active": 1}
    _, learners, h2 = train(
        num_steps=160_000, log_window=20_000,
        env_kwargs=env_s2, learners=learners, verbose=True,
    )
    print()
    print_policy(learners[TYPE_B], "B (after stage 2)")

    # =======================================================================
    # STAGE 3: joint, medium stream (4 of each)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: joint, target_active=4")
    print("=" * 70)
    for t in (TYPE_A, TYPE_B):
        _reset_schedule(learners[t],
                        alpha_start=0.08, alpha_min=0.02, alpha_decay_steps=250_000,
                        eps_start=0.5, eps_min=0.01, eps_decay_steps=250_000)
    env_s3 = {**base_env, "enabled_types": (TYPE_A, TYPE_B), "target_active": 4}
    _, learners, h3 = train(
        num_steps=250_000, log_window=20_000,
        env_kwargs=env_s3, learners=learners, verbose=True,
    )

    # =======================================================================
    # STAGE 4: joint, dense stream (6 of each)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: joint, target_active=6, low alpha/eps")
    print("=" * 70)
    for t in (TYPE_A, TYPE_B):
        _reset_schedule(learners[t],
                        alpha_start=0.02, alpha_min=0.02, alpha_decay_steps=1,
                        eps_start=0.01, eps_min=0.01, eps_decay_steps=1)
    env_s4 = {**base_env, "enabled_types": (TYPE_A, TYPE_B), "target_active": 6}
    _, learners, h4 = train(
        num_steps=120_000, log_window=20_000,
        env_kwargs=env_s4, learners=learners, verbose=True,
    )

    # =======================================================================
    # STAGE 5: fine-tune at very low rate
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 5: joint, very low alpha/eps fine-tune")
    print("=" * 70)
    for t in (TYPE_A, TYPE_B):
        _reset_schedule(learners[t],
                        alpha_start=0.01, alpha_min=0.01, alpha_decay_steps=1,
                        eps_start=0.005, eps_min=0.005, eps_decay_steps=1)
    env_s5 = {**base_env, "enabled_types": (TYPE_A, TYPE_B), "target_active": 6}
    _, learners, h5 = train(
        num_steps=600_000, log_window=20_000,
        env_kwargs=env_s5, learners=learners, verbose=True,
    )

    # =======================================================================
    # Final policies + rollouts
    # =======================================================================
    print()
    print_policy(learners[TYPE_A], "A FINAL")
    print()
    print_policy(learners[TYPE_B], "B FINAL")

    # Q-value inspection at critical states
    print("\n=== Critical-state Q-values ===")
    for label, t, pos, carry in [
        ("A heading to U, antechamber (2,1)", TYPE_A, (2, 1), 0),
        ("A returning to X, antechamber (2,3)", TYPE_A, (2, 3), 1),
        ("B heading to V, antechamber (1,2)", TYPE_B, (1, 2), 0),
        ("B returning to Y, antechamber (3,2)", TYPE_B, (3, 2), 1),
    ]:
        for lk in (0, 1):
            s = (pos[0], pos[1], carry, lk)
            q = learners[t].Q[s]
            best_val = max(q)
            best_acts = [i for i, v in enumerate(q) if v == best_val]
            names = ["N", "S", "E", "W", "WAIT"]
            print(f"  {label}, lake={'DRY' if lk == 0 else 'FLOOD'}:")
            print(f"    Q = {[round(v, 3) for v in q]}  (N,S,E,W,WAIT)")
            print(f"    best={'/'.join(names[i] for i in best_acts)}  unique={len(best_acts)==1}")

    print("\n=== Greedy rollout: lake DRY throughout (no flips) ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[0] * 25, start_lake=False, verbose=True)
    print("\n=== Greedy rollout: lake FLOODED throughout ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[1] * 25, start_lake=True, verbose=True)
    print("\n=== Greedy rollout: lake alternates each step ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[i % 2 for i in range(25)],
                   start_lake=False, verbose=True)

    plot_history(h5, out_path=os.path.join(OUT_DIR, "training_curves_stage5.png"))
    plot_history(h3, out_path=os.path.join(OUT_DIR, "training_curves_stage3.png"))
    for name, h in [("h1", h1), ("h2", h2), ("h3", h3), ("h4", h4), ("h5", h5)]:
        with open(os.path.join(OUT_DIR, f"history_{name}.json"), "w") as f:
            json.dump(h, f, indent=2, default=lambda o: None)

    print("\n=== Final summary (stage 5 tail) ===")
    last = {k: v[-1] for k, v in h5.items() if isinstance(v, list) and v}
    for k in ("collision_rate", "water_A_rate", "delivery_rate_A", "delivery_rate_B",
             "avg_return_A", "avg_return_B", "avg_steps_A", "avg_steps_B",
             "A_dry_cross_share", "B_flooded_cross_share"):
        print(f"  {k:>25s} = {last.get(k)}")


if __name__ == "__main__":
    main()
