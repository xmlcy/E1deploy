#!/usr/bin/env python3
"""Send raw JSON to the RL API without importing the SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from websockets.asyncio.client import connect


def command_for(args: argparse.Namespace) -> dict[str, object]:
    command: dict[str, object] = {
        "ctl_mode": "loco" if args.action != "status" else "none",
        "vel_x": 0.0,
        "vel_y": 0.0,
        "vel_yaw": 0.0,
        "base_roll": 0.0,
        "base_pitch": 0.0,
        "base_yaw": 0.0,
        "base_z": 0.0,
        "loco_mode": "loco_walk_and_stand",
        "arm_mode": "none",
    }
    if args.action == "walk":
        command.update(vel_x=args.vel_x, vel_y=args.vel_y, vel_yaw=args.vel_yaw)
    elif args.action == "stop":
        command["loco_mode"] = "loco_stop"
    elif args.action in ("dance1", "dance2"):
        command["dance_mode"] = args.action
    return command


async def run(args: argparse.Namespace) -> None:
    async with connect(args.uri, compression=None) as websocket:
        await websocket.send(json.dumps(args.auth_token))
        auth_response = await websocket.recv()
        if json.loads(auth_response) != "droidup":
            raise RuntimeError(f"authentication failed: {auth_response!r}")
        print(f"authenticated: {auth_response}")

        command = command_for(args)
        print("send:", json.dumps(command, ensure_ascii=False))
        await websocket.send(json.dumps(command))

        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            response = json.loads(await asyncio.wait_for(websocket.recv(), 2.0))
            label = "command failed" if "error" in response else "status"
            print(f"{label}:", json.dumps(response, ensure_ascii=False))

        if args.action == "walk":
            command.update(vel_x=0.0, vel_y=0.0, vel_yaw=0.0)
            await websocket.send(json.dumps(command))
            print("sent zero-velocity stand before disconnect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://localhost:8765")
    parser.add_argument("--auth-token", default="droidpad")
    parser.add_argument(
        "--action",
        choices=("status", "stand", "walk", "stop", "dance1", "dance2"),
        default="status",
    )
    parser.add_argument("--vel-x", type=float, default=0.2)
    parser.add_argument("--vel-y", type=float, default=0.0)
    parser.add_argument("--vel-yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
