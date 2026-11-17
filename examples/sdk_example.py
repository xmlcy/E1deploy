#!/usr/bin/env python3
"""Minimal synchronous SDK example with safe rejection handling."""

from __future__ import annotations

import signal
import time

from e1_rl_sdk import E1RobotClient


def exit_instead_of_suspend(signum, _frame) -> None:
    """Let the client context close its WebSocket instead of being suspended."""
    print("\nCtrl+Z detected: closing the WebSocket safely", flush=True)
    raise SystemExit(128 + signum)


if hasattr(signal, "SIGTSTP"):
    signal.signal(signal.SIGTSTP, exit_instead_of_suspend)


with E1RobotClient("ws://localhost:8765") as robot:
    status = robot.recv_status(timeout=2.0)
    print("真实状态:", status)

    result = robot.stand()
    if not result.success:
        print(f"站立命令未执行: {result.code}: {result.message}")
        print(
            "当前状态:",
            result.fsm_state,
            "活动策略:",
            result.active_policy,
            "恢复模式:",
            result.recovery_active,
        )
        print("若处于 damping，请先用手柄 X 进入默认姿态，角度正确后按 A。")
    else:
        print("站立命令已接受:", result)
        walk_result = robot.walk(vel_x=0.2)
        if not walk_result.success:
            print(f"行走命令未执行: {walk_result.code}: {walk_result.message}")
        else:
            print("行走命令已接受:", walk_result)
            robot.stand()
        # print(robot.recv_status(timeout=2.0))

        robot.walk(vel_x=1.5)
        # print(robot.recv_status(timeout=2.0))
        print("行走")
        time.sleep(10.0)
        robot.stand()
        print("站立")
        robot.dance(1, force=True)
        # print(robot.recv_status(timeout=2.0))
        print("舞蹈1")
        time.sleep(20.0)
        robot.dance(2, force=True)
        # print(robot.recv_status(timeout=2.0))
        print("舞蹈2")
        time.sleep(15.0)
        robot.stand()
        print("站立")
        # print(robot.recv_status(timeout=2.0))
