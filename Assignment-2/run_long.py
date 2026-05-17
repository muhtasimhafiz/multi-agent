"""Longer training run with stronger exploration & higher gamma.

Standard Q-learning with collision penalty bumped to -50 reflects the
"serious damage" wording in the spec and makes the traffic-light coordination
strictly preferable to "always cross" for Type B.
"""

import json
import os

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_long")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    env_kwargs = dict(
        p_lake_flip=0.1,
        spawn_prob=0.3,
        r_step=-5,
        r_wait=-3,
        r_collision=-50,   # stronger than the spec default to make coordination strictly optimal
        r_water=-20,
        r_pickup=10,
        r_deliver=50,
    )
    q_kwargs = dict(alpha=0.1, gamma=0.99, eps=1.0, eps_min=0.1,
                    eps_decay_steps=500_000)

    print("=== Long joint training ===")
    print("env:", env_kwargs)
    print("q:  ", q_kwargs)
    print()

    env, learners, history = train(
        num_steps=1_000_000,
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

    print("\n=== Greedy rollout: lake starts DRY, stays dry ===")
    greedy_rollout(learners, max_steps=20,
                   lake_pattern=[0] * 20, start_lake=False, verbose=True)

    print("\n=== Greedy rollout: lake starts FLOODED, stays flooded ===")
    greedy_rollout(learners, max_steps=20,
                   lake_pattern=[1] * 20, start_lake=True, verbose=True)

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
