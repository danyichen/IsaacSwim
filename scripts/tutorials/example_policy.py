"""A minimal policy for `isaacswim_bench.run --policy`; replace it with your own."""

from __future__ import annotations


class HoldPosition:
    def __init__(self, action):
        self.action = action

    def reset(self, instruction):
        self.instruction = instruction

    def __call__(self, observation):
        return self.action.copy()


def load(env):
    return HoldPosition(env.zero_action())
