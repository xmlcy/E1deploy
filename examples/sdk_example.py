#!/usr/bin/env python3
"""Minimal synchronous SDK example."""

from __future__ import annotations

import time

from e1_rl_sdk import E1RobotClient


with E1RobotClient("ws://localhost:8765") as robot:
    robot.stand()
    print(robot.recv_status(timeout=2.0))

    robot.walk(vel_x=0.5)
    print(robot.recv_status(timeout=2.0))
    time.sleep(10.0)
    robot.dance(1)
    print(robot.recv_status(timeout=2.0))
    time.sleep(6.0)
    robot.dance(2)
    print(robot.recv_status(timeout=2.0))
    time.sleep(6.0)
    robot.stand()
    print(robot.recv_status(timeout=2.0))
