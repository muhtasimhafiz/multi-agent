"""Sequential training with collision penalty -50.

Same two-phase protocol as run_sequential.py but with the stronger collision
penalty that the spec hints at with "serious damage". This makes
the traffic-light coordination strictly Q-dominant for B (gap > Q-table noise),
so the optimal policy should converge cleanly.
"""

import json
import os

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B
from qlearn import QLearner


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_seq_strong")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    base_env = dict(
        p_lake_flip=0.1,
        spawn_prob=0.15,
        r_step=-5,
        r_wait=-3,
        r_collision=-50,   # stronger than spec default; pushes traffic-light strictly Q-dominant
        r_water=-20,
        r_pickup=10,
        r_deliver=50,
    )
    q_kwargs = dict(
        q_init=50.0,
        alpha=0.2, alpha_min=0.01, alpha_decay_steps=400_000,
        gamma=0.99,
        eps=0.1, eps_min=0.05, eps_decay_steps=400_000,
    )

    print("=" * 70)
    print("PHASE 1: A alone (no B in env), collision=-50")
    print("=" * 70)
    env_phase1 = {**base_env, "enabled_types": (TYPE_A,)}
    _, learners, hist1 = train(
        num_steps=600_000,
        log_window=20_000,
        env_kwargs=env_phase1,
        q_kwargs_a=q_kwargs,
        q_kwargs_b=q_kwargs,
        verbose=True,
    )
    print()
    print_policy(learners[TYPE_A], "A")

    print("\n" + "=" * 70)
    print("PHASE 2: B vs frozen A, collision=-50")
    print("=" * 70)
    learners[TYPE_B] = QLearner(**q_kwargs)
    env_phase2 = {**base_env, "enabled_types": (TYPE_A, TYPE_B)}
    _, learners, hist2 = train(
        num_steps=900_000,
        log_window=20_000,
        env_kwargs=env_phase2,
        learners=learners,
        frozen_types=(TYPE_A,),
        verbose=True,
    )

    print()
    print_policy(learners[TYPE_A], "A (frozen)")
    print()
    print_policy(learners[TYPE_B], "B")

    print("\n=== Greedy rollout: lake DRY throughout ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[0] * 25, start_lake=False, verbose=True)
    print("\n=== Greedy rollout: lake FLOODED throughout ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[1] * 25, start_lake=True, verbose=True)
    print("\n=== Greedy rollout: lake alternates each step ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[i % 2 for i in range(25)],
                   start_lake=False, verbose=True)

    plot_history(hist2, out_path=os.path.join(OUT_DIR, "training_curves_phase2.png"))
    plot_history(hist1, out_path=os.path.join(OUT_DIR, "training_curves_phase1.png"))
    with open(os.path.join(OUT_DIR, "history_phase1.json"), "w") as f:
        json.dump(hist1, f, indent=2, default=lambda o: None)
    with open(os.path.join(OUT_DIR, "history_phase2.json"), "w") as f:
        json.dump(hist2, f, indent=2, default=lambda o: None)

    print("\n=== Phase 2 final summary ===")
    last = {k: v[-1] for k, v in hist2.items() if isinstance(v, list) and v}
    for k in ("collision_rate", "water_A_rate", "delivery_rate_A", "delivery_rate_B",
             "avg_return_A", "avg_return_B", "avg_steps_A", "avg_steps_B",
             "A_dry_cross_share", "B_flooded_cross_share"):
        print(f"  {k:>25s} = {last.get(k)}")


if __name__ == "__main__":
    main()
