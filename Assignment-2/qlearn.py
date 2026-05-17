"""
Tabular standard Q-learning for FIT5226 Stage 2 - Task 1.

  Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Two Q-tables (one shared across all Type-A robots, one across all Type-B
robots) - parameter sharing within a type. No expected-Bellman target and no
marginalisation over lake transitions (hint 4 is intentionally not applied):
the bootstrap uses the actually observed s'.
"""

import random
from collections import defaultdict

from env import NUM_ACTIONS


class QLearner:
    def __init__(self, alpha=0.1, gamma=0.95,
                 eps=1.0, eps_min=0.05, eps_decay_steps=150_000,
                 alpha_min=None, alpha_decay_steps=None):
        self.Q = defaultdict(lambda: [0.0] * NUM_ACTIONS)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self._eps_start = eps
        self._eps_decay_steps = max(1, eps_decay_steps)
        # alpha schedule (linear). If alpha_min is None, alpha stays constant.
        self._alpha_start = alpha
        self._alpha_min = alpha if alpha_min is None else alpha_min
        self._alpha_decay_steps = max(1, alpha_decay_steps or 1)
        self._steps = 0

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, NUM_ACTIONS - 1)
        return self._argmax(state)

    def greedy(self, state):
        return self._argmax(state)

    def _argmax(self, state):
        q = self.Q[state]
        m = max(q)
        # tie-break randomly among best actions
        best = [i for i, v in enumerate(q) if v == m]
        return random.choice(best)

    def update(self, s, a, r, s_next, done):
        q = self.Q[s]
        if done:
            target = r
        else:
            target = r + self.gamma * max(self.Q[s_next])
        q[a] += self.alpha * (target - q[a])

    def step_eps(self):
        # linear decay of eps and (independently) alpha
        self._steps += 1
        eps_frac = min(1.0, self._steps / self._eps_decay_steps)
        self.eps = self._eps_start + (self.eps_min - self._eps_start) * eps_frac
        alpha_frac = min(1.0, self._steps / self._alpha_decay_steps)
        self.alpha = self._alpha_start + (self._alpha_min - self._alpha_start) * alpha_frac
