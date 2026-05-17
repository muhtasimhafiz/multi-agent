"""
Evaluation utilities for FIT5226 Stage 2 - Task 1.

Provides:
  - greedy_rollout: deterministic single-robot rollouts under a frozen policy
    (one A and one B placed together in an env), reports path + metrics,
  - policy_grid: prints/returns the greedy action map for a type given the
    has_sample and lake slices,
  - plot_history: matplotlib plots of training curves (if matplotlib present).
"""

from env import (GRID, LAKE_POS, X_POS, Y_POS, U_POS, V_POS,
                 TYPE_A, TYPE_B, NUM_ACTIONS, ACTION_NAMES, apply_move,
                 home_of, target_of, Env, Robot)


def policy_grid(qlearner, has_sample, lake):
    """Return a 5x5 grid of greedy actions for a given (has_sample, lake) slice."""
    out = []
    for r in range(GRID):
        row = []
        for c in range(GRID):
            s = (r, c, int(has_sample), int(lake))
            a = qlearner.greedy(s)
            row.append(a)
        out.append(row)
    return out


def print_policy(qlearner, agent_type):
    print(f"=== Greedy policy for Type {agent_type} ===")
    for hs in (0, 1):
        for lk in (0, 1):
            grid = policy_grid(qlearner, hs, lk)
            label = f"has_sample={hs}, lake={'FLOODED' if lk else 'DRY'}"
            print(f"\n  {label}")
            for r in range(GRID):
                line = "    "
                for c in range(GRID):
                    cell_pos = (r, c)
                    marker = ACTION_NAMES[grid[r][c]]
                    # annotate fixed locations
                    if cell_pos == X_POS:
                        tag = "X"
                    elif cell_pos == Y_POS:
                        tag = "Y"
                    elif cell_pos == U_POS:
                        tag = "U"
                    elif cell_pos == V_POS:
                        tag = "V"
                    elif cell_pos == LAKE_POS:
                        tag = "L"
                    else:
                        tag = "."
                    line += f"{tag}{marker} "
                print(line)


def greedy_rollout(learners, max_steps=40, lake_pattern=None, start_lake=False, verbose=True):
    """Run one A and one B together under greedy policies and report the path.

    lake_pattern: optional list of booleans giving the lake state per step
                  (length max_steps). If None, lake stays at start_lake.
    """
    # Build a fresh env with NO new spawns and no stochastic lake unless asked.
    env = Env(spawn_prob=0.0, p_lake_flip=0.0)
    env.lake_flooded = start_lake
    # Place exactly one A at X and one B at Y.
    a = Robot(TYPE_A)
    a.pos = X_POS
    b = Robot(TYPE_B)
    b.pos = Y_POS
    env.robots = [a, b]

    def policy(robot, obs):
        return learners[robot.type].greedy(obs)

    history = []
    a_done = b_done = False
    a_steps = b_steps = 0
    a_collisions = b_collisions = 0
    a_water = 0

    for t in range(max_steps):
        if lake_pattern is not None and t < len(lake_pattern):
            env.lake_flooded = bool(lake_pattern[t])
        trs = env.tick(policy)
        for tr in trs:
            if tr["type"] == TYPE_A:
                if not a_done:
                    a_steps += 1
                if tr["collision"]:
                    a_collisions += 1
                if tr["water"]:
                    a_water += 1
                if tr["delivered"]:
                    a_done = True
            else:
                if not b_done:
                    b_steps += 1
                if tr["collision"]:
                    b_collisions += 1
                if tr["delivered"]:
                    b_done = True
        # snapshot positions for trace
        pa = a.pos if not a_done else "DONE"
        pb = b.pos if not b_done else "DONE"
        history.append({
            "t": t + 1,
            "lake": int(env.lake_flooded),
            "A_pos": pa,
            "B_pos": pb,
        })
        if a_done and b_done:
            break

    summary = {
        "A_steps_to_deliver": a_steps if a_done else None,
        "B_steps_to_deliver": b_steps if b_done else None,
        "A_collisions": a_collisions,
        "B_collisions": b_collisions,
        "A_water_hits": a_water,
        "trace": history,
    }
    if verbose:
        print(f"A delivered in {summary['A_steps_to_deliver']} steps "
              f"(collisions={a_collisions}, water_hits={a_water})")
        print(f"B delivered in {summary['B_steps_to_deliver']} steps "
              f"(collisions={b_collisions})")
        for h in history:
            print(f"  t={h['t']:>2d}  lake={'F' if h['lake'] else 'D'}  "
                  f"A={h['A_pos']}  B={h['B_pos']}")
    return summary


def plot_history(history, out_path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    s = history["step"]

    axes[0, 0].plot(s, history["collision_rate"], label="A-B collisions / step")
    axes[0, 0].plot(s, history["water_A_rate"], label="A in flooded lake / step")
    axes[0, 0].set_title("Safety metrics")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].legend()

    axes[0, 1].plot(s, history["delivery_rate_A"], label="A deliveries / step")
    axes[0, 1].plot(s, history["delivery_rate_B"], label="B deliveries / step")
    axes[0, 1].set_title("Throughput")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].legend()

    axes[1, 0].plot(s, history["avg_return_A"], label="A avg return / round-trip")
    axes[1, 0].plot(s, history["avg_return_B"], label="B avg return / round-trip")
    axes[1, 0].set_title("Average undiscounted return per delivered robot")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].legend()

    axes[1, 1].plot(s, history["avg_steps_A"], label="A avg steps / round-trip")
    axes[1, 1].plot(s, history["avg_steps_B"], label="B avg steps / round-trip")
    axes[1, 1].axhline(8, ls="--", color="gray", label="optimal=8")
    axes[1, 1].set_title("Steps per round-trip (lower is faster)")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].legend()

    axes[2, 0].plot(s, history["A_dry_cross_share"], label="P(lake=dry | A in lake)")
    axes[2, 0].plot(s, history["B_flooded_cross_share"], label="P(lake=flooded | B in lake)")
    axes[2, 0].set_title("Traffic-light coordination")
    axes[2, 0].set_xlabel("training step")
    axes[2, 0].set_ylim(0, 1.05)
    axes[2, 0].legend()

    axes[2, 1].plot(s, history["eps_A"], label="epsilon")
    axes[2, 1].set_title("Exploration schedule")
    axes[2, 1].set_xlabel("training step")
    axes[2, 1].legend()

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
        print(f"saved plot to {out_path}")
    return fig
