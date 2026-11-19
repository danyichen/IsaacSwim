from __future__ import annotations

import numpy as np

from ..core import Command, Expert, World
from ..platform import TOOL_GRASP, arm_pose_for, gripper_open, vehicle_for_tool

HOVER = 0.14
DESCEND_RATE = 0.04
CLOSE_HOLD = 1.8


class ToolExpert(Expert):
    DESCEND_RATE = DESCEND_RATE
    BOTTOM_SETTLE = 1.5

    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.arm_cmd: dict | None = None
        self.hold_pos: np.ndarray | None = None
        self.hold_yaw = 0.0
        self.descend_from = 0.0
        self.bottom_t: float | None = None

    def tool_base(self, world: World) -> np.ndarray:
        return self.arm.fk(world.arm_q)[0]

    def cmd_tool(self, world: World, want, grip: bool) -> Command:
        pos = vehicle_for_tool(want, self.tool_base(world), world.quat)
        self.hold_pos = pos
        return Command(pos=pos, yaw=self.hold_yaw, arm=self.arm_cmd, grip=grip)

    def cmd_hold(self, world: World, grip: bool) -> Command:
        if self.hold_pos is None:
            self.hold_pos = world.pos.copy()
        return Command(pos=self.hold_pos, yaw=self.hold_yaw, arm=self.arm_cmd, grip=grip)

    def plan_grasp_arm(self, world: World, close_across_w) -> dict:
        q = arm_pose_for(self.arm, TOOL_GRASP, seed=world.arm_q, quat=world.quat,
                         close_across_w=close_across_w)
        if q is None:
            raise RuntimeError(f"the working column {TOOL_GRASP} is unreachable")
        return q

    def arm_arrived(self, world: World, tol: float = 0.03) -> bool:
        return all(abs(world.arm_q[n] - v) < tol for n, v in self.arm_cmd.items())

    @staticmethod
    def jaws_open(world: World) -> bool:
        return gripper_open(world.grip_q)

    def over(self, world: World, want, tol_xy: float) -> bool:
        return float(np.linalg.norm((world.tool_pos - want)[:2])) < tol_xy

    def descend(self, world: World, target, floor: float, grip: bool = False):
        want = np.array([target[0], target[1],
                         max(floor, self.descend_from - self.DESCEND_RATE * self.state_t)])
        at_bottom = want[2] <= floor + 1e-6
        if at_bottom and self.bottom_t is None:
            self.bottom_t = self.state_t
        settled = (at_bottom and self.state_t - self.bottom_t > self.BOTTOM_SETTLE
                   and abs(world.tool_pos[2] - want[2]) < 0.009
                   and self.over(world, want, 0.013))
        return self.cmd_tool(world, want, grip), settled

    def start_descent(self, world: World, state: str) -> None:
        self.descend_from = float(world.tool_pos[2])
        self.bottom_t = None
        self.goto(state, world.t)
