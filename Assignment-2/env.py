"""
FIT5226 Stage 2 - Task 1 environment.

A 5x5 grid with two robot types (A from X to U, B from Y to V) running as a
continuous stream. The lake cell at the center is a coordination signal: its
state (dry/flooded) flips probabilistically each step. Type A is penalised for
being in the lake while it is flooded; A-B collisions in any cell are penalised.

Per-step sequence (matches the spec):
  1. all active robots observe the state
  2. all robots choose an action (handled by the policy passed to tick)
  3. all robots execute simultaneously (proposed new positions resolved)
  4. environment updates (pickups / deliveries) - lake unchanged
  5. rewards are computed using the current (pre-flip) lake state
  6. lake may flip with probability p_lake_flip
"""

import random

GRID = 5

NORTH, SOUTH, EAST, WEST, WAIT = 0, 1, 2, 3, 4
NUM_ACTIONS = 5
ACTION_NAMES = ["N", "S", "E", "W", "."]

X_POS = (2, 0)
Y_POS = (0, 2)
U_POS = (2, 4)
V_POS = (4, 2)
LAKE_POS = (2, 2)

TYPE_A = "A"
TYPE_B = "B"


def home_of(t):
    return X_POS if t == TYPE_A else Y_POS


def target_of(t):
    return U_POS if t == TYPE_A else V_POS


def apply_move(pos, action):
    r, c = pos
    if action == NORTH:
        nr, nc = r - 1, c
    elif action == SOUTH:
        nr, nc = r + 1, c
    elif action == EAST:
        nr, nc = r, c + 1
    elif action == WEST:
        nr, nc = r, c - 1
    else:
        return (r, c)
    if 0 <= nr < GRID and 0 <= nc < GRID:
        return (nr, nc)
    return (r, c)


class Robot:
    _next_id = 0

    def __init__(self, agent_type):
        self.id = Robot._next_id
        Robot._next_id += 1
        self.type = agent_type
        self.pos = home_of(agent_type)
        self.has_sample = False
        self.done = False
        self.born_step = None
        self.steps_alive = 0

    def obs(self, lake_flooded):
        return (self.pos[0], self.pos[1], int(self.has_sample), int(lake_flooded))


class Env:
    def __init__(self,
                 p_lake_flip=0.1,
                 spawn_prob=0.3,
                 r_step=-5,
                 r_wait=-3,
                 r_collision=-20,
                 r_water=-20,
                 r_pickup=10,
                 r_deliver=50):
        self.p_lake_flip = p_lake_flip
        self.spawn_prob = spawn_prob
        self.r_step = r_step
        self.r_wait = r_wait
        self.r_collision = r_collision
        self.r_water = r_water
        self.r_pickup = r_pickup
        self.r_deliver = r_deliver
        self.reset()

    def reset(self):
        self.lake_flooded = False
        self.robots = []
        self.t = 0

    def _home_free(self, agent_type):
        h = home_of(agent_type)
        return all(r.pos != h for r in self.robots)

    def maybe_spawn(self):
        # Robots are deployed at random times from each ship (continuous stream).
        # Only spawn if the home cell is free, so newly spawned robots don't
        # immediately collide with one still standing on it.
        if self._home_free(TYPE_A) and random.random() < self.spawn_prob:
            r = Robot(TYPE_A)
            r.born_step = self.t
            self.robots.append(r)
        if self._home_free(TYPE_B) and random.random() < self.spawn_prob:
            r = Robot(TYPE_B)
            r.born_step = self.t
            self.robots.append(r)

    def tick(self, policy):
        """Advance one global timestep.

        `policy(robot, obs)` returns an action in {0..4}.
        Returns a list of transition dicts: {type, id, s, a, r, s_next, done,
        collision, water, picked, delivered}.
        """
        # Step 0: continuous deployment.
        self.maybe_spawn()

        active = list(self.robots)
        if not active:
            # No active robots; still advance the lake clock.
            if random.random() < self.p_lake_flip:
                self.lake_flooded = not self.lake_flooded
            self.t += 1
            return []

        lake_now = self.lake_flooded

        # Step 1+2: observe + choose action.
        pre_obs = {r.id: r.obs(lake_now) for r in active}
        actions = {r.id: policy(r, pre_obs[r.id]) for r in active}

        # Step 3: execute simultaneously.
        for r in active:
            r.pos = apply_move(r.pos, actions[r.id])

        # Step 4: env updates (auto pickup / deliver flags).
        picked = {r.id: False for r in active}
        delivered = {r.id: False for r in active}
        for r in active:
            if not r.has_sample and r.pos == target_of(r.type):
                r.has_sample = True
                picked[r.id] = True
            if r.has_sample and r.pos == home_of(r.type):
                delivered[r.id] = True

        # Collisions: post-move cells containing BOTH a Type A and a Type B.
        cells = {}
        for r in active:
            cells.setdefault(r.pos, []).append(r)
        collision_cells = set()
        for pos, occupants in cells.items():
            ts = {rr.type for rr in occupants}
            if TYPE_A in ts and TYPE_B in ts:
                collision_cells.add(pos)

        # Step 5: rewards (lake state is still lake_now).
        rewards = {}
        water_hits = {}
        for r in active:
            a = actions[r.id]
            rew = self.r_wait if a == WAIT else self.r_step
            water_hit = (r.type == TYPE_A and r.pos == LAKE_POS and lake_now)
            if water_hit:
                rew += self.r_water
            water_hits[r.id] = water_hit
            if r.pos in collision_cells:
                rew += self.r_collision
            if picked[r.id]:
                rew += self.r_pickup
            if delivered[r.id]:
                rew += self.r_deliver
            rewards[r.id] = rew

        # Step 6: lake may flip.
        if random.random() < self.p_lake_flip:
            self.lake_flooded = not self.lake_flooded

        transitions = []
        for r in active:
            r.steps_alive += 1
            done = delivered[r.id]
            s_next = r.obs(self.lake_flooded)
            transitions.append({
                "type": r.type,
                "id": r.id,
                "s": pre_obs[r.id],
                "a": actions[r.id],
                "r": rewards[r.id],
                "s_next": s_next,
                "done": done,
                "collision": r.pos in collision_cells,
                "water": water_hits[r.id],
                "picked": picked[r.id],
                "delivered": delivered[r.id],
                "lake_at_step": int(lake_now),
                "pos_after": r.pos,
                "steps_alive": r.steps_alive,
            })
            if done:
                r.done = True

        # Retire delivered robots.
        self.robots = [r for r in self.robots if not r.done]
        self.t += 1
        return transitions
