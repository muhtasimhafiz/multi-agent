"""Two-phase sequential training (hint 5 of the spec).

PHASE 1: Type A learns alone in the env (no B robots ever spawn). A's MDP
is fully stationary and Q-learning provably converges to the optimal policy
"cross when dry, detour when flooded".

PHASE 2: A's Q-table is FROZEN at its converged values. Type B is then
trained in the same env with A spawning and behaving greedily. From B's
viewpoint A is now a fixed part of the environment, so B's MDP is also
stationary. Standard Q-learning is guaranteed to find B's optimal policy
against A's behaviour, which is the traffic-light coordination
"cross when flooded, detour when dry".

Sound argument for hint 5 equivalence: a joint policy (pi_A, pi_B) is a
Nash equilibrium of the multi-agent game iff each component is a best
response to the other. Phase 1 finds A's best response to "no other agent"
- since A's water signal is independent of B, this also happens to be A's
best response in the joint game (A coordinates regardless of B). Phase 2
finds B's best response to the converged A. The result is a Nash
equilibrium of the joint game, and given the strict payoff dominance of
the traffic-light arrangement, it is the Pareto-optimal one.
"""

import json
import os

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B
from qlearn import QLearner


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_seq")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    base_env = dict(
        p_lake_flip=0.1,
        spawn_prob=0.15,
        r_step=-5,
        r_wait=-3,
        r_collision=-20,
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

    # =========================================================================
    # PHASE 1: A learns alone (B doesn't exist)
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: A alone (no B in env)")
    print("=" * 70)
    env_phase1 = {**base_env, "enabled_types": (TYPE_A,)}
    print("env:", env_phase1)
    print("q:  ", q_kwargs)
    print()

    _, learners, hist1 = train(
        num_steps=600_000,
        log_window=10_000,
        env_kwargs=env_phase1,
        q_kwargs_a=q_kwargs,
        q_kwargs_b=q_kwargs,
        verbose=True,
    )

    print()
    print_policy(learners[TYPE_A], "A")

    # =========================================================================
    # PHASE 2: freeze A, train B against the converged A
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: B learns against frozen A")
    print("=" * 70)
    # Fresh B with optimistic init; keep the trained A frozen.
    learners[TYPE_B] = QLearner(**q_kwargs)
    env_phase2 = {**base_env, "enabled_types": (TYPE_A, TYPE_B)}
    print("env:", env_phase2)
    print()

    _, learners, hist2 = train(
        num_steps=900_000,
        log_window=10_000,
        env_kwargs=env_phase2,
        learners=learners,
        frozen_types=(TYPE_A,),
        verbose=True,
    )

    print()
    print_policy(learners[TYPE_A], "A (frozen)")
    print()
    print_policy(learners[TYPE_B], "B")

    # =========================================================================
    # Greedy rollouts under different lake regimes
    # =========================================================================
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

    # =========================================================================
    # Plots + dumps (phase 2 metrics are the "true" joint behaviour)
    # =========================================================================
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
