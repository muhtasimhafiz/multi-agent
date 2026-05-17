"""Tuned long run: alpha decay + slower eps decay + lighter spawn rate.

Goal: drive both Type A and Type B to the optimal traffic-light coordination
under the spec rewards using standard Q-learning on a continuous stream.
"""

import json
import os

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_tuned")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    env_kwargs = dict(
        p_lake_flip=0.1,
        spawn_prob=0.15,   # lighter stream -> less non-stationarity noise per visit
        r_step=-5,
        r_wait=-3,
        r_collision=-20,   # spec default
        r_water=-20,
        r_pickup=10,
        r_deliver=50,
    )
    # alpha 0.2 -> 0.01 over 600k steps; eps 1.0 -> 0.1 over 600k steps
    q_kwargs = dict(
        alpha=0.2, alpha_min=0.01, alpha_decay_steps=600_000,
        gamma=0.99,
        eps=1.0, eps_min=0.1, eps_decay_steps=600_000,
    )

    print("=== Tuned joint training (alpha + eps decay) ===")
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

    print("\n=== Greedy rollout: lake DRY ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[0] * 25, start_lake=False, verbose=True)

    print("\n=== Greedy rollout: lake FLOODED ===")
    greedy_rollout(learners, max_steps=25,
                   lake_pattern=[1] * 25, start_lake=True, verbose=True)

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
