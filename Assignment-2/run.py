"""
End-to-end driver for Task 1.

Runs joint training under a continuous stream of deployments, prints the
final greedy policies for both types, runs deterministic greedy rollouts to
verify the optimal coordinated paths emerge, and saves training-curve plots.
"""

import json
import os

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    env_kwargs = dict(
        p_lake_flip=0.1,
        spawn_prob=0.3,
        r_step=-5,
        r_wait=-3,
        r_collision=-20,
        r_water=-20,
        r_pickup=10,
        r_deliver=50,
    )
    q_kwargs = dict(alpha=0.1, gamma=0.97, eps=1.0, eps_min=0.05,
                    eps_decay_steps=200_000)

    print("=== Training (joint, continuous stream) ===")
    print("env:", env_kwargs)
    print("q:  ", q_kwargs)
    print()

    env, learners, history = train(
        num_steps=400_000,
        log_window=5_000,
        env_kwargs=env_kwargs,
        q_kwargs_a=q_kwargs,
        q_kwargs_b=q_kwargs,
        verbose=True,
    )

    print()
    print_policy(learners[TYPE_A], "A")
    print()
    print_policy(learners[TYPE_B], "B")

    # ------ Greedy rollouts (frozen policies, no exploration) ------
    print("\n=== Greedy rollout: lake starts DRY, stays dry ===")
    greedy_rollout(learners, max_steps=20,
                   lake_pattern=[0] * 20, start_lake=False, verbose=True)

    print("\n=== Greedy rollout: lake starts FLOODED, stays flooded ===")
    greedy_rollout(learners, max_steps=20,
                   lake_pattern=[1] * 20, start_lake=True, verbose=True)

    print("\n=== Greedy rollout: lake alternates each step ===")
    greedy_rollout(learners, max_steps=20,
                   lake_pattern=[i % 2 for i in range(20)],
                   start_lake=False, verbose=True)

    # ------ Plots + history dump ------
    plot_history(history, out_path=os.path.join(OUT_DIR, "training_curves.png"))
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=lambda o: None)

    # ------ Final summary stats ------
    print("\n=== Final summary ===")
    last = {k: v[-1] for k, v in history.items() if isinstance(v, list) and v}
    for k in ("collision_rate", "water_A_rate", "delivery_rate_A", "delivery_rate_B",
             "avg_return_A", "avg_return_B", "avg_steps_A", "avg_steps_B",
             "A_dry_cross_share", "B_flooded_cross_share"):
        print(f"  {k:>25s} = {last.get(k)}")


if __name__ == "__main__":
    main()
