"""Short-budget variant: 10k+10k+50k+100k = 170k total steps."""

import time
import random
from collections import defaultdict, deque

# Reuse my existing env/qlearn/train via the canonical modules
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from env import Env, TYPE_A, TYPE_B, LAKE_POS, X_POS, Y_POS
from qlearn import QLearner
from train import train
from eval import print_policy, greedy_rollout


def reset(lr, alpha, eps, eps_min, eps_decay_steps):
    lr.alpha = lr._alpha_start = alpha
    lr._alpha_min = alpha            # no within-stage alpha decay
    lr._alpha_decay_steps = 1
    lr.eps = lr._eps_start = eps
    lr.eps_min = eps_min
    lr._eps_decay_steps = max(1, eps_decay_steps)
    lr._steps = 0


def crit_table(learners):
    names = ['N', 'S', 'E', 'W', 'WAIT']
    cases = [
        ('A heading to U at (2,1)', TYPE_A, (2,1), 0),
        ('A returning to X at (2,3)', TYPE_A, (2,3), 1),
        ('B heading to V at (1,2)', TYPE_B, (1,2), 0),
        ('B returning to Y at (3,2)', TYPE_B, (3,2), 1),
    ]
    print('=== Critical-state Q-values ===')
    all_unique = True
    for label, t, pos, carry in cases:
        for lk in (0, 1):
            s = (pos[0], pos[1], carry, lk)
            q = learners[t].Q[s]
            mx = max(q)
            best = [i for i, v in enumerate(q) if v == mx]
            unique = len(best) == 1
            all_unique = all_unique and unique
            print(f'  {label}, lake={"DRY" if lk == 0 else "FLOOD"}: '
                  f'best={"/".join(names[i] for i in best)} unique={unique}  '
                  f'Q={[round(v, 2) for v in q]}')
    print(f'\nALL UNIQUE BEST: {all_unique}')
    return all_unique


def main():
    t0 = time.time()
    base = dict(p_lake_flip=0.5, r_step=-5, r_wait=-3,
                r_collision=-20, r_water=-20, r_pickup=10, r_deliver=50)

    learners = {
        TYPE_A: QLearner(q_init=0.0, alpha=0.15, alpha_min=0.15, alpha_decay_steps=1,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=10_000),
        TYPE_B: QLearner(q_init=0.0, alpha=0.15, alpha_min=0.15, alpha_decay_steps=1,
                         gamma=0.95, eps=1.0, eps_min=0.01, eps_decay_steps=10_000),
    }

    print('STAGE 1: A solo, 10k steps, alpha=0.15')
    _ = train(num_steps=10_000, log_window=2_500,
          env_kwargs={**base, 'enabled_types': (TYPE_A,), 'target_active': 1},
          learners=learners, verbose=True)

    print('\nSTAGE 2: B solo, 10k steps, alpha=0.15')
    reset(learners[TYPE_B], 0.15, 1.0, 0.01, 10_000)
    _ = train(num_steps=10_000, log_window=2_500,
          env_kwargs={**base, 'enabled_types': (TYPE_B,), 'target_active': 1},
          learners=learners, verbose=True)

    print('\nSTAGE 3: joint, 50k steps, target_active=4, alpha=0.05')
    for t in (TYPE_A, TYPE_B):
        reset(learners[t], 0.05, 0.5, 0.01, 50_000)
    _ = train(num_steps=50_000, log_window=10_000,
          env_kwargs={**base, 'enabled_types': (TYPE_A, TYPE_B), 'target_active': 4},
          learners=learners, verbose=True)

    print('\nSTAGE 5: fine-tune, 100k steps, alpha=0.01')
    for t in (TYPE_A, TYPE_B):
        reset(learners[t], 0.01, 0.005, 0.005, 1)
    _, _, h5 = train(num_steps=100_000, log_window=20_000,
               env_kwargs={**base, 'enabled_types': (TYPE_A, TYPE_B), 'target_active': 4},
               learners=learners, verbose=True)

    print(f'\n=== Wall clock: {time.time()-t0:.1f}s ===\n')

    print_policy(learners[TYPE_A], 'A')
    print()
    print_policy(learners[TYPE_B], 'B')
    print()
    all_unique = crit_table(learners)
    print()

    print('=== Rollout: lake locked DRY ===')
    greedy_rollout(learners, max_steps=12, lake_pattern=[0]*12, verbose=True)
    print('\n=== Rollout: lake locked FLOODED ===')
    greedy_rollout(learners, max_steps=12, lake_pattern=[1]*12, start_lake=True, verbose=True)

    print('\n=== Final stage-5 tail metrics ===')
    last = {k: v[-1] for k, v in h5.items() if isinstance(v, list) and v}
    for k in ('collision_rate', 'delivery_rate_A', 'delivery_rate_B',
              'avg_return_A', 'avg_return_B', 'avg_steps_A', 'avg_steps_B',
              'A_dry_cross_share', 'B_flooded_cross_share'):
        print(f'  {k:>25s} = {last.get(k)}')


if __name__ == '__main__':
    main()
