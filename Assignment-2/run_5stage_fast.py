"""Fast variant of the 5-stage curriculum.

Cuts:
- Stages 1+2 shortened (A and B converge on individual MDPs quickly)
- Stage 4 removed (Stage 3 + Stage 5 already cover dense+fine-tune)
- Stage 5 cut from 600k to 150k (enough for stable greedy policy, may
  leave some Q-ties resolved by random tie-break rather than strictly)

Trade-off: greedy policy may rely on tie-breaking at a few states.
With random tie-break this still produces traffic-light behaviour on
average but slightly more eval-run-to-eval-run variance.
"""

import json, os, time

from train import train
from eval import print_policy, greedy_rollout, plot_history
from env import TYPE_A, TYPE_B
from qlearn import QLearner


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_5stage_fast")
os.makedirs(OUT_DIR, exist_ok=True)


def _reset(lr, a, am, ad, e, em, ed):
    lr.alpha = lr._alpha_start = a; lr._alpha_min = am; lr._alpha_decay_steps = max(1, ad)
    lr.eps = lr._eps_start = e; lr.eps_min = em; lr._eps_decay_steps = max(1, ed)
    lr._steps = 0


def main():
    t0 = time.time()
    base_env = dict(p_lake_flip=0.5, r_step=-5, r_wait=-3, r_collision=-20,
                    r_water=-20, r_pickup=10, r_deliver=50)
    learners = {
        TYPE_A: QLearner(q_init=0.0, alpha=0.2, alpha_min=0.02, alpha_decay_steps=100_000,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=100_000),
        TYPE_B: QLearner(q_init=0.0, alpha=0.2, alpha_min=0.02, alpha_decay_steps=100_000,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=100_000),
    }

    # ---- STAGE 1: A solo (was 200k -> 100k) ----
    print("STAGE 1: A solo, 100k")
    _, learners, _ = train(num_steps=100_000, log_window=25_000,
                           env_kwargs={**base_env, "enabled_types": (TYPE_A,), "target_active": 1},
                           learners=learners, verbose=True)

    # ---- STAGE 2: B solo (was 160k -> 80k) ----
    print("\nSTAGE 2: B solo, 80k")
    _reset(learners[TYPE_B], 0.2, 0.02, 80_000, 1.0, 0.01, 80_000)
    _, learners, _ = train(num_steps=80_000, log_window=20_000,
                           env_kwargs={**base_env, "enabled_types": (TYPE_B,), "target_active": 1},
                           learners=learners, verbose=True)

    # ---- STAGE 3: joint (was 250k -> 150k) ----
    print("\nSTAGE 3: joint target_active=4, 150k")
    for t in (TYPE_A, TYPE_B):
        _reset(learners[t], 0.08, 0.02, 150_000, 0.5, 0.01, 150_000)
    _, learners, h3 = train(num_steps=150_000, log_window=25_000,
                            env_kwargs={**base_env, "enabled_types": (TYPE_A, TYPE_B), "target_active": 4},
                            learners=learners, verbose=True)

    # ---- STAGE 5: short fine-tune (skip stage 4, cut 5 from 600k -> 100k) ----
    print("\nSTAGE 5: low-eps fine-tune target_active=4, 100k")
    for t in (TYPE_A, TYPE_B):
        _reset(learners[t], 0.01, 0.01, 1, 0.005, 0.005, 1)
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
        ("A at (2,1)", TYPE_A, (2, 1), 0),
        ("B at (1,2)", TYPE_B, (1, 2), 0),
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
    greedy_rollout(learners, max_steps=12, lake_pattern=[0]*12, start_lake=False, verbose=False)
    print("\n=== Rollout: lake FLOODED ===")
    greedy_rollout(learners, max_steps=12, lake_pattern=[1]*12, start_lake=True, verbose=False)

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
