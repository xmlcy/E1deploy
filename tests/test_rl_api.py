from __future__ import annotations

import asyncio
import json
import time
import unittest

from websockets.sync.client import connect

from e1_rl_sdk import (
    AsyncE1RobotClient,
    E1RobotClient,
    RobotCommand as SdkRobotCommand,
)
from rl_api import ProtocolError, RobotCommand, RobotStatus, RlWebSocketServer


class ProtocolTests(unittest.TestCase):
    def test_documented_command_round_trip(self) -> None:
        payload = {
            "ctl_mode": "loco",
            "vel_x": 0.5,
            "vel_y": 0.0,
            "vel_yaw": -0.2,
            "base_roll": 0.0,
            "base_pitch": 0.0,
            "base_yaw": 0.0,
            "base_z": 0.0,
            "loco_mode": "loco_walk_and_stand",
            "arm_mode": "none",
        }
        command = RobotCommand.from_payload(payload)
        self.assertEqual(command.vel_x, 0.5)
        self.assertEqual(command.dance_mode, "none")
        self.assertEqual(command.to_dict(include_extensions=False), payload)

    def test_rejects_response_only_loco_fall(self) -> None:
        with self.assertRaises(ProtocolError):
            RobotCommand.from_payload(
                {"ctl_mode": "loco", "loco_mode": "loco_fall"}
            )

    def test_dance_requires_zero_velocity(self) -> None:
        with self.assertRaises(ProtocolError):
            RobotCommand.from_payload(
                {
                    "ctl_mode": "loco",
                    "loco_mode": "loco_walk_and_stand",
                    "vel_x": 0.1,
                    "dance_mode": "dance1",
                }
            )


class ServerAndSdkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = RlWebSocketServer("127.0.0.1", 0, status_hz=50.0)
        self.server.publish_status(
            RobotStatus(
                bus_voltage=48.2,
                joint_q=tuple(float(i) for i in range(10)),
                joint_dq=(0.0,) * 10,
                loco_mode="loco_walk_and_stand",
                fsm_state="hybrid",
                active_policy="hybrid",
                can_accept_velocity=True,
                can_accept_dance=True,
            )
        )
        self.server.start()
        self.uri = f"ws://127.0.0.1:{self.server.bound_port}"

    def tearDown(self) -> None:
        self.server.stop()

    def test_auth_status_command_and_disconnect_returns_to_gamepad(self) -> None:
        with E1RobotClient(self.uri) as client:
            status = client.recv_status(timeout=1.0)
            self.assertAlmostEqual(status.bus_voltage, 48.2)
            self.assertEqual(status.fsm_state, "hybrid")
            self.assertEqual(status.active_policy, "hybrid")
            self.assertTrue(status.can_accept_velocity)
            self.assertTrue(status.can_accept_dance)
            self.assertFalse(status.recovery_active)
            client.walk(0.3, -0.1, 0.2)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                command, _ = self.server.command_snapshot()
                if command.vel_x == 0.3:
                    break
                time.sleep(0.01)
            self.assertEqual(command.ctl_mode, "loco")
            self.assertEqual(command.vel_x, 0.3)

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            command, _ = self.server.command_snapshot()
            if command == RobotCommand():
                break
            time.sleep(0.01)
        self.assertEqual(command, RobotCommand())

    def test_dance_extension_is_queued(self) -> None:
        with E1RobotClient(self.uri) as client:
            client.dance(2)
            deadline = time.monotonic() + 1.0
            dance = None
            while time.monotonic() < deadline and dance is None:
                dance = self.server.pop_dance_request()
                if dance is None:
                    time.sleep(0.01)
            self.assertEqual(dance, "dance2")

            self.server.release_control()
            command, _ = self.server.command_snapshot()
            self.assertEqual(command.ctl_mode, "none")
            self.assertIsNone(self.server.pop_dance_request())

    def test_repeated_dance_packet_is_only_one_edge(self) -> None:
        with E1RobotClient(self.uri) as client:
            command = SdkRobotCommand.dance(1)
            client.send_command(command)
            deadline = time.monotonic() + 1.0
            dance = None
            while time.monotonic() < deadline and dance is None:
                dance = self.server.pop_dance_request()
                if dance is None:
                    time.sleep(0.01)
            self.assertEqual(dance, "dance1")

            # Releasing ownership must not re-arm a client that continues to
            # stream the same dance packet.
            self.server.release_control()
            client.send_command(command)
            time.sleep(0.05)
            self.assertIsNone(self.server.pop_dance_request())

            # An explicit non-dance packet creates the next rising edge.
            client.release()
            client.send_command(command)
            deadline = time.monotonic() + 1.0
            dance = None
            while time.monotonic() < deadline and dance is None:
                dance = self.server.pop_dance_request()
                if dance is None:
                    time.sleep(0.01)
            self.assertEqual(dance, "dance1")

    def test_invalid_actual_state_returns_failure(self) -> None:
        self.server.publish_status(
            RobotStatus(
                joint_q=(0.0,) * 10,
                joint_dq=(0.0,) * 10,
                loco_mode="loco_walk_and_stand",
                fsm_state="mimic",
                active_policy="victory1",
                dance_mode="dance1",
                control_source="policy",
                can_accept_velocity=False,
                can_accept_dance=False,
            )
        )
        with connect(self.uri, compression=None) as websocket:
            websocket.send(json.dumps("droidpad"))
            self.assertEqual(json.loads(websocket.recv(timeout=1.0)), "droidup")
            websocket.send(json.dumps(SdkRobotCommand.dance(2).to_dict()))
            deadline = time.monotonic() + 1.0
            error = None
            while time.monotonic() < deadline and error is None:
                payload = json.loads(websocket.recv(timeout=1.0))
                error = payload.get("error")
            self.assertEqual(error["code"], "invalid_state")
            self.assertEqual(error["fsm_state"], "mimic")
            self.assertEqual(error["dance_mode"], "dance1")

        command, _ = self.server.command_snapshot()
        self.assertEqual(command.ctl_mode, "none")

    def test_async_sdk(self) -> None:
        async def scenario() -> None:
            async with AsyncE1RobotClient(self.uri) as client:
                await client.stand()
                status = await asyncio.wait_for(client.recv_status(), 1.0)
                self.assertEqual(status.loco_mode, "loco_walk_and_stand")

        asyncio.run(scenario())

    def test_bad_authentication_is_closed(self) -> None:
        with connect(self.uri, compression=None) as websocket:
            websocket.send(json.dumps("wrong-token"))
            with self.assertRaises(Exception):
                websocket.recv(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
