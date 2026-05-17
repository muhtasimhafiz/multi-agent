"""Joint training in a collision-rich environment.

Maintains 4 active A's and 4 active B's at all times (env target_active=4).
The 5x5 grid is then very crowded - especially the lake cell, which is the
only column-2 / row-2 crossing of both shortest paths. Collisions become
frequent enough that the traffic-light coordination is unambiguously the
highest-value joint policy for both types, and the gradient toward it is
strong even for independent Q-learners.

This is still STANDARD Q-learning, joint training, continuous stream, no
seeds, hint 4 ignored - just with more agents present at once.
"""

import json
import os

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_dense")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    env_kwargs = dict(
        p_lake_flip=0.1,
        target_active=4,           # always 4 of each type active
        r_step=-5,
        r_wait=-3,
        r_collision=-20,           # spec default
        r_water=-20,
        r_pickup=10,
        r_deliver=50,
    )
    q_kwargs = dict(
        q_init=50.0,
        alpha=0.2, alpha_min=0.01, alpha_decay_steps=800_000,
        gamma=0.99,
        eps=0.2, eps_min=0.05, eps_decay_steps=800_000,
    )

    print("=== Dense joint training (4 of each type, continuous stream) ===")
    print("env:", env_kwargs)
    print("q:  ", q_kwargs)
    print()

    env, learners, history = train(
        num_steps=1_500_000,
        log_window=10_000,
        env_kwargs=env_kwargs,
        q_kwargs_a=q_kwargs,
        q_kwargs_b=q_kwargs,
        verbose=True,
    )

    print()
    print_policy(learners[TYPE_A], "A")
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

    plot_history(history, out_path=os.path.join(OUT_DIR, "training_curves.png"))
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=lambda o: None)

    print("\n=== Final summary ===")
    last = {k: v[-1] for k, v in history.items() if isinstance(v, list) and v}
    for k in ("collision_rate", "water_A_rate", "delivery_rate_A", "delivery_rate_B",
             "avg_return_A", "avg_return_B", "avg_steps_A", "avg_steps_B",
             "A_dry_cross_share", "B_flooded_cross_share"):
        print(f"  {k:>25s} = {last.get(k)}")


if __name__ == "__main__":
    main()
