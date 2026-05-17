"""5-stage curriculum with NO alpha decay (alpha fixed at 0.1 throughout).

Same protocol as run_5stage.py but every stage uses a CONSTANT alpha = 0.1.
The alpha schedule from the working solution decays 0.2 -> 0.02 across
stages, with the late stages relying on very-low alpha to drive WAIT from
"tied" to "uniquely best" at antechamber states. Hypothesis: with alpha
fixed at 0.1, the noise floor is permanently a 10-sample EMA, so even if
traffic-light coordination emerges in policy, the Q-values may stay tied
or flicker at the marginal states.
"""

import json, os, time

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B
from qlearn import QLearner


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_5stage_noalpha")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.1   # fixed throughout


def _reset(lr, e, em, ed):
    """Reset only the epsilon schedule. Alpha stays at ALPHA (no decay)."""
    lr.alpha = lr._alpha_start = ALPHA
    lr._alpha_min = ALPHA
    lr._alpha_decay_steps = 1  # irrelevant since start==min
    lr.eps = lr._eps_start = e
    lr.eps_min = em
    lr._eps_decay_steps = max(1, ed)
    lr._steps = 0


def main():
    t0 = time.time()
    base_env = dict(p_lake_flip=0.5, r_step=-5, r_wait=-3, r_collision=-20,
                    r_water=-20, r_pickup=10, r_deliver=50)
    # Both learners use alpha=ALPHA fixed throughout
    learners = {
        TYPE_A: QLearner(q_init=0.0, alpha=ALPHA, alpha_min=ALPHA, alpha_decay_steps=1,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=100_000),
        TYPE_B: QLearner(q_init=0.0, alpha=ALPHA, alpha_min=ALPHA, alpha_decay_steps=1,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=100_000),
    }

    # ---- STAGE 1: A solo ----
    print(f"STAGE 1: A solo (alpha={ALPHA} fixed)")
    _, learners, _ = train(num_steps=100_000, log_window=25_000,
                           env_kwargs={**base_env, "enabled_types": (TYPE_A,), "target_active": 1},
                           learners=learners, verbose=True)

    # ---- STAGE 2: B solo ----
    print(f"\nSTAGE 2: B solo (alpha={ALPHA} fixed)")
    _reset(learners[TYPE_B], 1.0, 0.01, 80_000)
    _, learners, _ = train(num_steps=80_000, log_window=20_000,
                           env_kwargs={**base_env, "enabled_types": (TYPE_B,), "target_active": 1},
                           learners=learners, verbose=True)

    # ---- STAGE 3: joint medium ----
    print(f"\nSTAGE 3: joint target_active=4 (alpha={ALPHA} fixed)")
    for t in (TYPE_A, TYPE_B):
        _reset(learners[t], 0.5, 0.01, 150_000)
    _, learners, h3 = train(num_steps=150_000, log_window=25_000,
                            env_kwargs={**base_env, "enabled_types": (TYPE_A, TYPE_B), "target_active": 4},
                            learners=learners, verbose=True)

    # ---- STAGE 5: fine-tune (low eps only; alpha still fixed) ----
    print(f"\nSTAGE 5: low-eps fine-tune (alpha={ALPHA} fixed)")
    for t in (TYPE_A, TYPE_B):
        _reset(learners[t], 0.005, 0.005, 1)
    _, learners, h5 = train(num_steps=100_000, log_window=25_000,
                            env_kwargs={**base_env, "enabled_types": (TYPE_A, TYPE_B), "target_active": 4},
                            learners=learners, verbose=True)

    elapsed = time.time() - t0
    print(f"\n=== Total wall clock: {elapsed:.1f}s ===")

    print()
    print_policy(learners[TYPE_A], "A FINAL")
    print()
    print_policy(learners[TYPE_B], "B FINAL")

    print("\n=== Critical-state Q-values ===")
    for label, t, pos, carry in [
        ("A at (2,1) outbound", TYPE_A, (2, 1), 0),
        ("A at (2,3) returning", TYPE_A, (2, 3), 1),
        ("B at (1,2) outbound", TYPE_B, (1, 2), 0),
        ("B at (3,2) returning", TYPE_B, (3, 2), 1),
    ]:
        for lk in (0, 1):
            s = (pos[0], pos[1], carry, lk)
            q = learners[t].Q[s]
            mx = max(q)
            best = [i for i, v in enumerate(q) if v == mx]
            names = ["N", "S", "E", "W", "WAIT"]
            print(f"  {label} lake={'DRY' if lk==0 else 'FLOOD'}:  "
                  f"best={'/'.join(names[i] for i in best)} unique={len(best)==1}  "
                  f"Q={[round(v,2) for v in q]}")

    print("\n=== Rollout: lake DRY ===")
    greedy_rollout(learners, max_steps=12, lake_pattern=[0]*12, start_lake=False, verbose=True)
    print("\n=== Rollout: lake FLOODED ===")
    greedy_rollout(learners, max_steps=12, lake_pattern=[1]*12, start_lake=True, verbose=True)

    last = {k: v[-1] for k, v in h5.items() if isinstance(v, list) and v}
    print("\n=== Stage 5 tail ===")
    for k in ("collision_rate", "delivery_rate_A", "delivery_rate_B",
              "avg_return_A", "avg_return_B",
              "A_dry_cross_share", "B_flooded_cross_share"):
        print(f"  {k:>25s} = {last.get(k)}")

    plot_history(h5, out_path=os.path.join(OUT_DIR, "training_curves_stage5.png"))
    with open(os.path.join(OUT_DIR, "history_h5.json"), "w") as f:
        json.dump(h5, f, indent=2, default=lambda o: None)


if __name__ == "__main__":
    main()
