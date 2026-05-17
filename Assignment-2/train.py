"""
Joint training loop for FIT5226 Stage 2 - Task 1.

Both Type A and Type B robots learn in parallel on the same continuous stream
of deployments. Each timestep:
  - any free home cell may spawn a new robot,
  - every active robot acts (epsilon-greedy from its type's shared Q-table),
  - the env resolves moves, pickups, deliveries, collisions, water damage,
  - the lake may flip,
  - each robot's (s, a, r, s', done) is fed back into its type's Q-table.
"""

from collections import deque, defaultdict

from env import Env, TYPE_A, TYPE_B, LAKE_POS, X_POS, Y_POS, U_POS, V_POS
from qlearn import QLearner


def train(num_steps=300_000,
          log_window=2000,
          env_kwargs=None,
          q_kwargs_a=None,
          q_kwargs_b=None,
          verbose=True,
          learners=None,
          frozen_types=()):
    """Joint training loop.

    learners: optional pre-built {TYPE_A: QLearner, TYPE_B: QLearner}. If
        provided, q_kwargs_* are ignored and the existing tables continue.
    frozen_types: iterable of type strings whose Q-tables are NOT updated.
        Frozen agents still act epsilon-greedily from their (fixed) Q-table.
        Use e.g. frozen_types=(TYPE_A,) for phase 2 of sequential training.
    """
    env = Env(**(env_kwargs or {}))
    if learners is None:
        learners = {
            TYPE_A: QLearner(**(q_kwargs_a or {})),
            TYPE_B: QLearner(**(q_kwargs_b or {})),
        }
    frozen_types = set(frozen_types)
    # Force frozen agents to act greedily (no exploration) so they behave as a
    # fixed, deterministic environment for the learner(s).
    for t in frozen_types:
        learners[t].eps = 0.0
        learners[t].eps_min = 0.0
        learners[t]._eps_start = 0.0

    def policy(robot, obs):
        return learners[robot.type].act(obs)

    # rolling counters over the last log_window steps
    win = {
        "collisions": deque(maxlen=log_window),
        "water_A": deque(maxlen=log_window),
        "deliveries_A": deque(maxlen=log_window),
        "deliveries_B": deque(maxlen=log_window),
        "return_A": deque(maxlen=log_window),
        "return_B": deque(maxlen=log_window),
        "A_in_lake_when_flooded": deque(maxlen=log_window),
        "A_in_lake_when_dry": deque(maxlen=log_window),
        "B_in_lake_when_flooded": deque(maxlen=log_window),
        "B_in_lake_when_dry": deque(maxlen=log_window),
    }

    # per-robot accumulated reward, so a delivery's contribution to "return"
    # is the discounted-sum-style episode return (undiscounted here).
    robot_return = defaultdict(float)
    robot_steps = defaultdict(int)

    history = {
        "step": [],
        "collision_rate": [],
        "water_A_rate": [],
        "delivery_rate_A": [],
        "delivery_rate_B": [],
        "avg_return_A": [],
        "avg_return_B": [],
        "avg_steps_A": [],
        "avg_steps_B": [],
        "A_dry_cross_share": [],
        "B_flooded_cross_share": [],
        "eps_A": [],
        "eps_B": [],
    }

    delivered_returns_A = deque(maxlen=200)
    delivered_returns_B = deque(maxlen=200)
    delivered_steps_A = deque(maxlen=200)
    delivered_steps_B = deque(maxlen=200)

    for step in range(1, num_steps + 1):
        transitions = env.tick(policy)
        # update Q-tables, accumulate metrics
        n_coll = 0
        n_water_A = 0
        n_deliv_A = 0
        n_deliv_B = 0
        for tr in transitions:
            if tr["type"] not in frozen_types:
                learners[tr["type"]].update(tr["s"], tr["a"], tr["r"], tr["s_next"], tr["done"])
            rid = tr["id"]
            robot_return[rid] += tr["r"]
            robot_steps[rid] += 1
            if tr["collision"]:
                n_coll += 1
            if tr["water"]:
                n_water_A += 1
            # whether this robot is sitting in the lake cell after its move,
            # broken down by lake state at the moment of the step.
            in_lake = (tr["pos_after"] == LAKE_POS)
            if tr["type"] == TYPE_A:
                win["A_in_lake_when_flooded"].append(1 if (in_lake and tr["lake_at_step"]) else 0)
                win["A_in_lake_when_dry"].append(1 if (in_lake and not tr["lake_at_step"]) else 0)
            else:
                win["B_in_lake_when_flooded"].append(1 if (in_lake and tr["lake_at_step"]) else 0)
                win["B_in_lake_when_dry"].append(1 if (in_lake and not tr["lake_at_step"]) else 0)
            if tr["delivered"]:
                if tr["type"] == TYPE_A:
                    n_deliv_A += 1
                    delivered_returns_A.append(robot_return[rid])
                    delivered_steps_A.append(robot_steps[rid])
                else:
                    n_deliv_B += 1
                    delivered_returns_B.append(robot_return[rid])
                    delivered_steps_B.append(robot_steps[rid])
                del robot_return[rid]
                del robot_steps[rid]

        win["collisions"].append(n_coll)
        win["water_A"].append(n_water_A)
        win["deliveries_A"].append(n_deliv_A)
        win["deliveries_B"].append(n_deliv_B)

        for t, lr in learners.items():
            if t in frozen_types:
                continue
            lr.step_eps()

        if step % log_window == 0:
            coll_rate = sum(win["collisions"]) / max(1, len(win["collisions"]))
            water_rate = sum(win["water_A"]) / max(1, len(win["water_A"]))
            dA = sum(win["deliveries_A"]) / max(1, len(win["deliveries_A"]))
            dB = sum(win["deliveries_B"]) / max(1, len(win["deliveries_B"]))

            A_fl = sum(win["A_in_lake_when_flooded"])
            A_dr = sum(win["A_in_lake_when_dry"])
            B_fl = sum(win["B_in_lake_when_flooded"])
            B_dr = sum(win["B_in_lake_when_dry"])
            A_share = A_dr / max(1, A_fl + A_dr)
            B_share = B_fl / max(1, B_fl + B_dr)

            avg_ret_A = (sum(delivered_returns_A) / len(delivered_returns_A)) if delivered_returns_A else float("nan")
            avg_ret_B = (sum(delivered_returns_B) / len(delivered_returns_B)) if delivered_returns_B else float("nan")
            avg_stp_A = (sum(delivered_steps_A) / len(delivered_steps_A)) if delivered_steps_A else float("nan")
            avg_stp_B = (sum(delivered_steps_B) / len(delivered_steps_B)) if delivered_steps_B else float("nan")

            history["step"].append(step)
            history["collision_rate"].append(coll_rate)
            history["water_A_rate"].append(water_rate)
            history["delivery_rate_A"].append(dA)
            history["delivery_rate_B"].append(dB)
            history["avg_return_A"].append(avg_ret_A)
            history["avg_return_B"].append(avg_ret_B)
            history["avg_steps_A"].append(avg_stp_A)
            history["avg_steps_B"].append(avg_stp_B)
            history["A_dry_cross_share"].append(A_share)
            history["B_flooded_cross_share"].append(B_share)
            history["eps_A"].append(learners[TYPE_A].eps)
            history["eps_B"].append(learners[TYPE_B].eps)

            if verbose:
                print(f"step={step:>7d}  eps={learners[TYPE_A].eps:.3f}  "
                      f"a={learners[TYPE_A].alpha:.3f}  "
                      f"coll/step={coll_rate:.4f}  "
                      f"A-in-flood={water_rate:.4f}  "
                      f"deliv A={dA:.3f} B={dB:.3f}  "
                      f"avg_steps A={avg_stp_A:.2f} B={avg_stp_B:.2f}  "
                      f"avg_return A={avg_ret_A:.2f} B={avg_ret_B:.2f}  "
                      f"A-dry={A_share:.2f} B-flood={B_share:.2f}")

    return env, learners, history


if __name__ == "__main__":
    env, learners, history = train(num_steps=300_000)
