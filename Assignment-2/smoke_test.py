"""Quick sanity checks for env mechanics."""
import random
from env import (Env, Robot, TYPE_A, TYPE_B, NORTH, SOUTH, EAST, WEST, WAIT,
                 X_POS, Y_POS, U_POS, V_POS, LAKE_POS, apply_move, GRID)


def test_apply_move():
    assert apply_move((2, 2), NORTH) == (1, 2)
    assert apply_move((2, 2), SOUTH) == (3, 2)
    assert apply_move((2, 2), EAST) == (2, 3)
    assert apply_move((2, 2), WEST) == (2, 1)
    assert apply_move((2, 2), WAIT) == (2, 2)
    # out of grid stays put
    assert apply_move((0, 0), NORTH) == (0, 0)
    assert apply_move((4, 4), SOUTH) == (4, 4)
    print("apply_move ok")


def test_pickup_deliver():
    env = Env(spawn_prob=0.0, p_lake_flip=0.0)
    a = Robot(TYPE_A); a.pos = (2, 3)
    env.robots = [a]

    def policy(r, o):
        return EAST  # one step east -> reaches U
    trs = env.tick(policy)
    assert a.pos == U_POS
    assert a.has_sample is True
    assert trs[0]["picked"] is True
    assert trs[0]["r"] == -5 + 10  # step + pickup
    # next: head back west
    def policy2(r, o):
        return WEST
    for _ in range(4):
        trs = env.tick(policy2)
    # After 4 westward steps from U we should be back at X with delivery
    assert trs[0]["delivered"] is True
    print("pickup/deliver ok; final reward incl. deliver:", trs[0]["r"])


def test_water_damage():
    env = Env(spawn_prob=0.0, p_lake_flip=0.0)
    env.lake_flooded = True
    a = Robot(TYPE_A); a.pos = (2, 1)
    env.robots = [a]
    def p(r, o): return EAST
    trs = env.tick(p)
    # A entered flooded lake -> step + water
    assert a.pos == LAKE_POS
    assert trs[0]["water"] is True
    assert trs[0]["r"] == -5 - 20
    # next step depart east; per spec, departing the lake -> no water penalty
    trs = env.tick(p)
    assert a.pos == (2, 3)
    assert trs[0]["water"] is False
    assert trs[0]["r"] == -5
    print("water damage ok")


def test_collision():
    env = Env(spawn_prob=0.0, p_lake_flip=0.0)
    env.lake_flooded = False
    a = Robot(TYPE_A); a.pos = (2, 1)
    b = Robot(TYPE_B); b.pos = (1, 2)
    env.robots = [a, b]
    # A goes east into lake, B goes south into lake -> collision
    def p(r, o):
        return EAST if r.type == TYPE_A else SOUTH
    trs = env.tick(p)
    by_type = {tr["type"]: tr for tr in trs}
    assert by_type[TYPE_A]["collision"] is True
    assert by_type[TYPE_B]["collision"] is True
    # A also dry-stepped lake, so no water hit
    assert by_type[TYPE_A]["water"] is False
    print("collision ok; rewards A,B =", by_type[TYPE_A]["r"], by_type[TYPE_B]["r"])


def test_same_type_no_collision():
    env = Env(spawn_prob=0.0, p_lake_flip=0.0)
    a1 = Robot(TYPE_A); a1.pos = (2, 1)
    a2 = Robot(TYPE_A); a2.pos = (3, 2)
    env.robots = [a1, a2]
    def p(r, o):
        return EAST if r.id == a1.id else NORTH
    trs = env.tick(p)
    for tr in trs:
        assert tr["collision"] is False
    print("same-type no collision ok")


if __name__ == "__main__":
    test_apply_move()
    test_pickup_deliver()
    test_water_damage()
    test_collision()
    test_same_type_no_collision()
    print("\nAll smoke tests passed.")
