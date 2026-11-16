"""Unified 50-Hz E1-21DOF Hybrid AMP / Mimic real-robot FSM.

IDLE is deliberately read-only: launching this process does not overwrite the
controller that was active before it. X claims control, A starts Hybrid, and
only a stationary Hybrid state may enter either Mimic policy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import from this directory deliberately. Importing ``fsm.*`` here would load
# the obsolete IsaacLab deployment under ``sim2real/fsm`` instead.
from hybrid_amp_inference import (  # noqa: E402
    REAL_JOINT_NAMES,
    HybridAmpInference,
)
from mimic_inference import MimicInference  # noqa: E402
from rl_api import RobotCommand, RobotStatus, RlWebSocketServer  # noqa: E402
from rl_api.protocol import finite_or_zero  # noqa: E402


class ControlState(Enum):
    IDLE = auto()
    GO_TO_DEFAULT = auto()
    STANDBY = auto()
    GO_TO_MIMIC_START = auto()
    GO_TO_HYBRID_START = auto()
    HYBRID = auto()
    MIMIC = auto()
    DAMPING = auto()


class UnifiedE1Fsm:
    dt = 0.02
    # A long open-loop pose interpolation cannot actively reject disturbances.
    # Keep mode transitions short, then let the selected policy take over.
    interpolation_duration = 0.2
    fall_tilt_threshold_deg = 50.0
    fall_hard_tilt_threshold_deg = 75.0
    fall_detection_frames = 3
    recovery_upright_tilt_deg = 30.0
    leg_damping = 4.0
    arm_damping = 2.0

    @property
    def state(self) -> ControlState:
        return self._state

    @state.setter
    def state(self, value: ControlState) -> None:
        previous = getattr(self, "_state", None)
        self._state = value
        if previous != value:
            previous_name = previous.name if previous is not None else "UNINITIALIZED"
            self.logger.info("STATE %s -> %s", previous_name, value.name)

    def __init__(
        self,
        hybrid_policy: Path,
        mimic1_policy: Path,
        mimic2_policy: Path,
        mimic1_motion_frames: int | None,
        mimic2_motion_frames: int | None,
        mimic1_start_frame: int,
        mimic2_start_frame: int,
        *,
        api_host: str = "0.0.0.0",
        api_port: int = 8765,
        api_auth_token: str = "droidpad",
        api_status_hz: float = 20.0,
        api_enabled: bool = True,
    ) -> None:
        self.logger = logging.getLogger("e1_fsm")
        # Load and validate all models before opening the robot connection.
        # Only the active model is evaluated inside the 50-Hz loop.
        self.hybrid = HybridAmpInference(hybrid_policy)
        self.mimics = {
            "victory1": MimicInference(
                mimic1_policy,
                mimic1_motion_frames,
                mimic1_start_frame,
                REAL_JOINT_NAMES,
            ),
            "victory2": MimicInference(
                mimic2_policy,
                mimic2_motion_frames,
                mimic2_start_frame,
                REAL_JOINT_NAMES,
            ),
        }
        self.active_mimic_name: str | None = None
        self.active_mimic: MimicInference | None = None
        if self.hybrid.dt != self.dt or self.hybrid.decimation != 1:
            raise ValueError("Hybrid inference must remain dt=0.02, decimation=1")
        for name, mimic in self.mimics.items():
            if mimic.dt != self.dt or mimic.decimation != 1:
                raise ValueError(
                    f"{name} inference must remain dt=0.02, decimation=1"
                )

        # Keep policy validation usable on a development machine without the
        # real-robot SDK.  Hardware modules are imported only when this runner
        # is actually constructed.
        from gamepad_controller import GamepadController
        from base.Base import NanoSleep
        from base.RobotBase import RobotBase
        from base.configs import E1_21dof as RobotConfig

        self.NanoSleep = NanoSleep
        print("[INFO] Initializing gamepad...")
        self.gamepad = GamepadController(deadzone=0.15)
        print("[INFO] Establishing gRPC hardware connection (Leg + Arm)...")
        self.robot = RobotBase(RobotConfig)
        self.robot.legActions = 13
        self.robot.armActions = 8
        while len(self.robot.legCommand.position) < 13:
            self.robot.legCommand.position.append(0.0)
            self.robot.legCommand.kp.append(0.0)
            self.robot.legCommand.kd.append(0.0)
        while len(self.robot.armCommand.position) < 8:
            self.robot.armCommand.position.append(0.0)
            self.robot.armCommand.kp.append(0.0)
            self.robot.armCommand.kd.append(0.0)
        print("[SUCCESS] Hardware communication initialized")

        self.state = ControlState.IDLE
        self.control_claimed = False
        self.target_q = np.zeros(21, dtype=np.float32)
        self.transition_start = np.zeros(21, dtype=np.float32)
        self.transition_target = np.zeros(21, dtype=np.float32)
        self.transition_kp = self.hybrid.kp_real.copy()
        self.transition_kd = self.hybrid.kd_real.copy()
        self.transition_elapsed = 0.0
        self.fall_counter = 0
        self.fall_latched = False
        self.recovery_default_active = False
        self.pending_api_dance: str | None = None
        self.hybrid_control_source = "gamepad"
        self.last_status_log_time = 0.0
        self.last_gamepad_log: tuple[object, ...] | None = None
        self.api_server = (
            RlWebSocketServer(
                api_host,
                api_port,
                auth_token=api_auth_token,
                status_hz=api_status_hz,
            )
            if api_enabled
            else None
        )

    def _joint_positions(self) -> np.ndarray:
        return np.asarray(
            list(self.robot.legState.position[:13])
            + list(self.robot.armState.position[:8]),
            dtype=np.float32,
        )

    def _send_position(
        self, target: np.ndarray, kp: np.ndarray, kd: np.ndarray
    ) -> None:
        for index in range(13):
            self.robot.legCommand.position[index] = float(target[index])
            self.robot.legCommand.kp[index] = float(kp[index])
            self.robot.legCommand.kd[index] = float(kd[index])
        for arm_index in range(8):
            index = 13 + arm_index
            self.robot.armCommand.position[arm_index] = float(target[index])
            self.robot.armCommand.kp[arm_index] = float(kp[index])
            self.robot.armCommand.kd[arm_index] = float(kd[index])
        self.robot.set_robot_command()

    def _send_damping(self) -> None:
        for index in range(13):
            self.robot.legCommand.position[index] = self.robot.legState.position[index]
            self.robot.legCommand.kp[index] = 0.0
            self.robot.legCommand.kd[index] = self.leg_damping
        for index in range(8):
            self.robot.armCommand.position[index] = self.robot.armState.position[index]
            self.robot.armCommand.kp[index] = 0.0
            self.robot.armCommand.kd[index] = self.arm_damping
        self.robot.set_robot_command()

    @staticmethod
    def _quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
        w = float(quat[0])
        xyz = quat[1:4]
        return (
            vector * (2.0 * w * w - 1.0)
            - 2.0 * w * np.cross(xyz, vector)
            + 2.0 * xyz * np.dot(xyz, vector)
        )

    def _torso_tilt_deg(self) -> float:
        euler = np.asarray(self.robot.legState.imu_euler[:3], dtype=np.float64)
        if euler.shape != (3,) or not np.all(np.isfinite(euler)):
            return float("inf")
        euler[euler > math.pi] -= 2.0 * math.pi
        roll, pitch, yaw = (float(value) for value in euler)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        quat = np.asarray(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dtype=np.float64,
        )
        gravity = self._quat_rotate_inverse(
            quat, np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
        )
        norm = float(np.linalg.norm(gravity))
        if norm < 1.0e-6 or not np.all(np.isfinite(gravity)):
            return float("inf")
        return math.degrees(math.acos(float(np.clip(-gravity[2] / norm, -1.0, 1.0))))

    def _hybrid_entry_ready(self) -> tuple[bool, float]:
        tilt = self._torso_tilt_deg()
        return (
            math.isfinite(tilt) and tilt <= self.recovery_upright_tilt_deg,
            tilt,
        )

    def _begin_transition(
        self,
        destination: ControlState,
        target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        self.transition_start = self._joint_positions()
        self.transition_target = np.asarray(target, dtype=np.float32).copy()
        self.transition_kp = np.asarray(kp, dtype=np.float32).copy()
        self.transition_kd = np.asarray(kd, dtype=np.float32).copy()
        self.transition_elapsed = 0.0
        self.state = destination
        self.control_claimed = True

    def _begin_default_transition(self) -> None:
        self._begin_transition(
            ControlState.GO_TO_DEFAULT,
            self.hybrid.default_pos_real,
            self.hybrid.kp_real,
            self.hybrid.kd_real,
        )

    def _begin_mimic_transition(self, name: str) -> None:
        mimic = self.mimics[name]
        self.active_mimic_name = name
        self.active_mimic = mimic
        self.hybrid.command_vel.fill(0.0)
        self._begin_transition(
            ControlState.GO_TO_MIMIC_START,
            mimic.start_pose_real,
            mimic.kp_real,
            mimic.kd_real,
        )

    def _enter_damping(self, reason: str, *, fall: bool = False) -> None:
        self.state = ControlState.DAMPING
        self.control_claimed = True
        self.fall_counter = 0
        self.fall_latched = fall
        self.recovery_default_active = False
        self.hybrid.command_vel.fill(0.0)
        self._send_damping()
        self.logger.warning("DAMPING reason=%s fall=%s", reason, fall)
        print(f"\n[SAFETY] {reason}: entered DAMPING")
        print("[SAFETY] X: move to default pose for assisted lifting; B: exit")

    def _check_fall(self) -> None:
        if self.recovery_default_active and self.state in (
            ControlState.GO_TO_DEFAULT,
            ControlState.STANDBY,
        ):
            # Keep commanding the known default joint pose while an operator
            # lifts the robot. Re-entering Hybrid remains angle-gated by A.
            self.fall_counter = 0
            return
        monitored = {
            ControlState.GO_TO_DEFAULT,
            ControlState.STANDBY,
            ControlState.GO_TO_MIMIC_START,
            ControlState.GO_TO_HYBRID_START,
            ControlState.HYBRID,
            ControlState.MIMIC,
        }
        if self.state not in monitored:
            self.fall_counter = 0
            return
        tilt = self._torso_tilt_deg()
        if not math.isfinite(tilt) or tilt >= self.fall_hard_tilt_threshold_deg:
            self.fall_counter = self.fall_detection_frames
        elif tilt >= self.fall_tilt_threshold_deg:
            self.fall_counter += 1
        else:
            self.fall_counter = 0
        if self.fall_counter >= self.fall_detection_frames:
            text = f"fall detected ({tilt:.1f} deg)" if math.isfinite(tilt) else "invalid IMU"
            self._enter_damping(text, fall=True)

    def _run_transition(self) -> None:
        progress = min(self.transition_elapsed / self.interpolation_duration, 1.0)
        blend_start = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.target_q = (
            blend_start * self.transition_start
            + (1.0 - blend_start) * self.transition_target
        ).astype(np.float32)
        self._send_position(
            self.target_q, self.transition_kp, self.transition_kd
        )
        self.transition_elapsed += self.dt

        if self.transition_elapsed < self.interpolation_duration:
            return
        self.target_q = self.transition_target.copy()
        if self.state == ControlState.GO_TO_DEFAULT:
            self.state = ControlState.STANDBY
            if self.recovery_default_active:
                print(
                    "[SAFETY] Recovery default pose reached: lift robot upright, "
                    "then press A"
                )
            else:
                print("[SUCCESS] Default pose reached: press A for Hybrid")
        elif self.state == ControlState.GO_TO_MIMIC_START:
            if self.active_mimic is None or self.active_mimic_name is None:
                raise RuntimeError("Mimic transition completed without a selected policy")
            self.active_mimic.reset()
            self.state = ControlState.MIMIC
            print(
                f"[SUCCESS] {self.active_mimic_name} start pose reached: "
                "Mimic inference started"
            )
        elif self.state == ControlState.GO_TO_HYBRID_START:
            self.hybrid.reset(self.robot)
            self.hybrid_control_source = "gamepad"
            self.state = ControlState.HYBRID
            print("[SUCCESS] Hybrid default pose reached: walk-run inference started")

    def _api_loco_mode(self) -> str:
        if self.fall_latched:
            return "loco_fall"
        if self.state == ControlState.DAMPING:
            return "loco_stop"
        if self.state == ControlState.IDLE:
            return "none"
        return "loco_walk_and_stand"

    def _api_active_policy(self) -> str:
        if self.state == ControlState.HYBRID:
            return "hybrid"
        if self.state in (ControlState.GO_TO_MIMIC_START, ControlState.MIMIC):
            return self.active_mimic_name or "mimic"
        return "none"

    def _api_dance_mode(self) -> str:
        if self.state not in (ControlState.GO_TO_MIMIC_START, ControlState.MIMIC):
            return "none"
        if self.active_mimic_name == "victory1":
            return "dance1"
        if self.active_mimic_name == "victory2":
            return "dance2"
        return "none"

    def _api_control_source(self) -> str:
        if self.state in (ControlState.GO_TO_MIMIC_START, ControlState.MIMIC):
            return "policy"
        if self.state == ControlState.HYBRID:
            return self.hybrid_control_source
        return "fsm"

    def _publish_api_status(self) -> None:
        # The interface document fixes these arrays at 10 elements. Select the
        # five primary pitch/roll/yaw/knee/ankle-pitch joints from each leg and
        # omit both ankle-roll joints plus the waist.
        main_joint_indices = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
        leg_position = self.robot.legState.position
        leg_velocity = self.robot.legState.velocity
        com_velocity = list(getattr(self.robot.legState, "est_com_vel", (0.0,) * 3))
        com_velocity.extend((0.0,) * (3 - len(com_velocity)))
        measured_velocity = tuple(finite_or_zero(value) for value in com_velocity[:3])
        gyro = list(getattr(self.robot.legState, "imu_gyro", (0.0,) * 3))
        gyro.extend((0.0,) * (3 - len(gyro)))
        measured_stationary = (
            math.hypot(measured_velocity[0], measured_velocity[1]) < 0.15
            and abs(measured_velocity[2]) < 0.20
            and abs(finite_or_zero(gyro[2])) < 0.30
        )
        upright = self._torso_tilt_deg() <= self.recovery_upright_tilt_deg
        status = RobotStatus(
            bus_voltage=finite_or_zero(getattr(self.robot.legState, "bus_voltage", 0.0)),
            bus_current=finite_or_zero(getattr(self.robot.legState, "bus_current", 0.0)),
            bus_energy=finite_or_zero(getattr(self.robot.legState, "bus_energy", 0.0)),
            vel_x=measured_velocity[0],
            vel_y=measured_velocity[1],
            vel_z=measured_velocity[2],
            joint_q=tuple(finite_or_zero(leg_position[index]) for index in main_joint_indices),
            joint_dq=tuple(finite_or_zero(leg_velocity[index]) for index in main_joint_indices),
            loco_mode=self._api_loco_mode(),
            fsm_state=self.state.name.lower(),
            active_policy=self._api_active_policy(),
            dance_mode=self._api_dance_mode(),
            control_source=self._api_control_source(),
            can_accept_velocity=self.state == ControlState.HYBRID,
            can_accept_dance=(
                self.state == ControlState.HYBRID
                and not self.hybrid.is_moving()
                and measured_stationary
                and upright
            ),
            transitioning=self.state in (
                ControlState.GO_TO_DEFAULT,
                ControlState.GO_TO_MIMIC_START,
                ControlState.GO_TO_HYBRID_START,
            ),
            recovery_active=self.recovery_default_active,
        )
        if self.api_server is not None:
            self.api_server.publish_status(status)
        now = time.monotonic()
        if now - self.last_status_log_time >= 1.0:
            self.last_status_log_time = now
            self.logger.info(
                "ROBOT_STATUS fsm_state=%s payload=%s",
                self.state.name,
                json.dumps(status.to_dict(), separators=(",", ":")),
            )

    def run(self) -> None:
        print("\n" + "=" * 66)
        print("E1-21DOF Unified Sim2Real FSM (50 Hz)")
        print("  startup : IDLE/read-only; no damping command is sent")
        print("  [X]     : damping -> default pose / assisted lifting")
        print("  [A]     : upright STANDBY -> Hybrid; Mimic -> Hybrid")
        print("  [START] : stationary Hybrid -> victory1 (button 11)")
        print("  [BACK]  : stationary Hybrid -> victory2 (button 10)")
        print("  Mimic1/Mimic2 cannot switch directly; press A first")
        print("  [Y]     : damping")
        print("  [B]     : damping and exit")
        if self.api_server is not None:
            print(
                f"  WebSocket API: ws://{self.api_server.host}:"
                f"{self.api_server.port}"
            )
        print("=" * 66 + "\n")

        timer = self.NanoSleep(self.dt * 1000.0)
        previous = {
            "a": False,
            "x": False,
            "y": False,
            "start": False,
            "back": False,
        }
        stats_start = time.perf_counter()
        loop_steps = 0
        policy_steps = 0
        active_policy = "none"

        try:
            if self.api_server is not None:
                self.api_server.start()
                print(
                    f"[SUCCESS] WebSocket API listening on "
                    f"{self.api_server.host}:{self.api_server.bound_port}"
                )
            while True:
                step_start = time.perf_counter()
                self.robot.get_robot_state()
                self.gamepad.update()

                buttons = {
                    "a": self.gamepad.get_button_a(),
                    "x": self.gamepad.get_button_x(),
                    "y": self.gamepad.get_button_y(),
                    "start": self.gamepad.get_button_start(),
                    "back": self.gamepad.get_button_back(),
                }
                pressed = {
                    name: value and not previous[name]
                    for name, value in buttons.items()
                }
                previous = buttons
                raw_gamepad_command = self.gamepad.get_commands()
                gamepad_motion = any(
                    abs(value) > 1.0e-6 for value in raw_gamepad_command
                )
                gamepad_b = self.gamepad.get_button_b()
                active_buttons = tuple(
                    name for name, value in {**buttons, "b": gamepad_b}.items()
                    if value
                )
                gamepad_log = (
                    *(round(float(value), 4) for value in raw_gamepad_command),
                    active_buttons,
                )
                if gamepad_log != self.last_gamepad_log:
                    self.last_gamepad_log = gamepad_log
                    self.logger.info(
                        "GAMEPAD_COMMAND vel_x=%.4f vel_y=%.4f vel_yaw=%.4f "
                        "buttons=%s",
                        raw_gamepad_command[0],
                        raw_gamepad_command[1],
                        raw_gamepad_command[2],
                        ",".join(active_buttons) or "none",
                    )
                manual_takeover_requested = (
                    gamepad_motion or any(pressed.values()) or gamepad_b
                )

                api_command = RobotCommand()
                if self.api_server is not None:
                    api_command, _ = self.api_server.command_snapshot()
                    dance_request = self.api_server.pop_dance_request()
                    if dance_request is not None:
                        self.pending_api_dance = dance_request
                api_loco_active = api_command.ctl_mode == "loco"

                # Local controls always have priority. This also revokes a
                # zero-velocity stand left behind by a disconnected client and
                # prevents a latched loco_stop from cancelling X recovery on
                # the following frame.
                if api_loco_active and manual_takeover_requested:
                    if self.api_server is not None:
                        self.api_server.release_control()
                    api_command = RobotCommand()
                    api_loco_active = False
                    self.pending_api_dance = None
                    self.hybrid_control_source = "gamepad"
                    print("[INFO] Gamepad input: WebSocket control released")
                    self.logger.info("CONTROL_OWNER websocket -> gamepad")

                if (
                    api_loco_active
                    and api_command.loco_mode == "loco_stop"
                    and self.state != ControlState.DAMPING
                ):
                    self.pending_api_dance = None
                    self._enter_damping("WebSocket loco_stop requested")

                if gamepad_b:
                    print("\n[EMERGENCY] B pressed: damping and exit")
                    self.logger.critical("EMERGENCY_STOP source=gamepad_b action=exit")
                    self.control_claimed = True
                    self._send_damping()
                    time.sleep(0.1)
                    break

                if pressed["y"] and self.state != ControlState.DAMPING:
                    self._enter_damping(f"Y pressed in {self.state.name}")

                self._check_fall()

                if self.state == ControlState.IDLE:
                    # Intentionally do not call set_robot_command() here.
                    if pressed["x"]:
                        print("[INFO] X pressed: claiming control and going to default pose")
                        self._begin_default_transition()

                elif self.state == ControlState.DAMPING:
                    if pressed["x"]:
                        self.recovery_default_active = True
                        self.logger.info(
                            "RECOVERY_DEFAULT_START source=gamepad_x "
                            "fall_latched=%s",
                            self.fall_latched,
                        )
                        print(
                            "[SAFETY] X pressed: moving to default pose; "
                            "lift robot upright before pressing A"
                        )
                        self._begin_default_transition()
                    else:
                        self._send_damping()

                elif self.state in (
                    ControlState.GO_TO_DEFAULT,
                    ControlState.GO_TO_MIMIC_START,
                    ControlState.GO_TO_HYBRID_START,
                ):
                    transition_state = self.state
                    self._run_transition()
                    if (
                        transition_state == ControlState.GO_TO_MIMIC_START
                        and self.state == ControlState.MIMIC
                    ):
                        active_policy = self.active_mimic_name or "mimic"
                        stats_start = time.perf_counter()
                        loop_steps = 0
                        policy_steps = 0
                    elif (
                        transition_state == ControlState.GO_TO_HYBRID_START
                        and self.state == ControlState.HYBRID
                    ):
                        active_policy = "hybrid"
                        stats_start = time.perf_counter()
                        loop_steps = 0
                        policy_steps = 0

                elif self.state == ControlState.STANDBY:
                    if pressed["a"]:
                        ready, tilt = self._hybrid_entry_ready()
                        if not ready:
                            print(
                                "[SAFETY] A ignored: robot is not upright "
                                f"(tilt={tilt:.1f} deg, required<="
                                f"{self.recovery_upright_tilt_deg:.1f} deg)"
                            )
                            self.logger.warning(
                                "HYBRID_ENTRY_REJECTED tilt_deg=%.3f threshold_deg=%.3f",
                                tilt,
                                self.recovery_upright_tilt_deg,
                            )
                            self.target_q = self.hybrid.default_pos_real.copy()
                            self._send_position(
                                self.target_q,
                                self.hybrid.kp_real,
                                self.hybrid.kd_real,
                            )
                        else:
                            self.recovery_default_active = False
                            self.fall_latched = False
                            self.hybrid.reset(self.robot)
                            self.target_q = self.hybrid.default_pos_real.copy()
                            self.hybrid_control_source = "gamepad"
                            self.state = ControlState.HYBRID
                            active_policy = "hybrid"
                            stats_start = time.perf_counter()
                            loop_steps = 0
                            policy_steps = 0
                            self.logger.info(
                                "HYBRID_ENTRY_ACCEPTED tilt_deg=%.3f", tilt
                            )
                            print(
                                "[SUCCESS] Hybrid walk-run inference started "
                                f"(tilt={tilt:.1f} deg)"
                            )
                    elif pressed["x"]:
                        self._begin_default_transition()
                    else:
                        self.target_q = self.hybrid.default_pos_real.copy()
                        self._send_position(
                            self.target_q,
                            self.hybrid.kp_real,
                            self.hybrid.kd_real,
                        )

                elif self.state == ControlState.HYBRID:
                    api_velocity = (
                        api_command.vel_x,
                        api_command.vel_y,
                        api_command.vel_yaw,
                    )
                    if api_loco_active:
                        self.hybrid.set_velocity_command(*api_velocity)
                        self.hybrid_control_source = "websocket"
                    else:
                        self.hybrid.set_gamepad_command(*raw_gamepad_command)
                        self.hybrid_control_source = "gamepad"
                    if pressed["start"] and pressed["back"]:
                        print("[INFO] START+BACK ignored: select only one Mimic policy")
                    elif pressed["start"] or pressed["back"] or self.pending_api_dance:
                        if self.hybrid.is_moving():
                            print(
                                "[INFO] Mimic request ignored: release the sticks and "
                                "wait for a zero velocity command before Mimic"
                            )
                        else:
                            api_dance = self.pending_api_dance
                            mimic_name = (
                                "victory1"
                                if pressed["start"] or api_dance == "dance1"
                                else "victory2"
                            )
                            button_name = (
                                "START"
                                if pressed["start"]
                                else "BACK" if pressed["back"] else "WebSocket"
                            )
                            self.pending_api_dance = None
                            print(
                                f"[INFO] {button_name} pressed in stationary Hybrid: "
                                f"moving to {mimic_name} start pose"
                            )
                            self._begin_mimic_transition(mimic_name)
                            self.logger.info(
                                "DANCE_START name=%s source=%s",
                                mimic_name,
                                button_name,
                            )
                            if api_dance is not None and self.api_server is not None:
                                # Dance is an edge-triggered action, not a
                                # persistent velocity-control lease. Releasing
                                # here prevents it from replaying after a fall
                                # recovery and lets the gamepad take over.
                                self.api_server.release_control()
                                api_command = RobotCommand()
                                api_loco_active = False
                                print(
                                    "[INFO] WebSocket dance accepted: "
                                    "control returned to gamepad"
                                )
                                self.logger.info(
                                    "CONTROL_OWNER websocket -> gamepad "
                                    "reason=dance_started"
                                )
                    elif pressed["x"]:
                        print("[INFO] Leaving Hybrid: returning to default pose")
                        self.hybrid.command_vel.fill(0.0)
                        self._begin_default_transition()
                    else:
                        try:
                            self.target_q = (
                                self.hybrid.step_velocity(self.robot, api_velocity)
                                if api_loco_active
                                else self.hybrid.step(
                                    self.robot, raw_gamepad_command
                                )
                            )
                            policy_steps += 1
                            self._send_position(
                                self.target_q,
                                self.hybrid.kp_real,
                                self.hybrid.kd_real,
                            )
                        except (FloatingPointError, RuntimeError, ValueError) as exc:
                            self._enter_damping(f"Hybrid inference failed: {exc}")

                elif self.state == ControlState.MIMIC:
                    if api_loco_active or self.pending_api_dance is not None:
                        rejected = self.pending_api_dance or api_command.dance_mode
                        print(
                            "[INFO] Stale WebSocket command dropped during Mimic"
                        )
                        self.logger.warning(
                            "COMMAND_DROPPED state=MIMIC command=%s reason=invalid_state",
                            rejected,
                        )
                        self.pending_api_dance = None
                        if self.api_server is not None:
                            self.api_server.release_control()
                        api_loco_active = False
                    if pressed["a"]:
                        print(
                            f"[INFO] Leaving {self.active_mimic_name}: "
                            "moving to Hybrid default pose"
                        )
                        self.logger.info(
                            "DANCE_EXIT name=%s source=gamepad_a",
                            self.active_mimic_name,
                        )
                        self._begin_transition(
                            ControlState.GO_TO_HYBRID_START,
                            self.hybrid.default_pos_real,
                            self.hybrid.kp_real,
                            self.hybrid.kd_real,
                        )
                    elif pressed["x"]:
                        print(
                            "[INFO] X ignored in Mimic: press A to return to "
                            "Hybrid before using X"
                        )
                        self.logger.warning(
                            "COMMAND_REJECTED state=MIMIC command=gamepad_x "
                            "reason=press_A_first"
                        )
                    elif pressed["start"] or pressed["back"]:
                        print(
                            "[INFO] Direct Mimic switching is disabled: "
                            "press A to return to Hybrid first"
                        )
                        self.logger.warning(
                            "COMMAND_REJECTED state=MIMIC command=gamepad_dance "
                            "reason=press_A_first"
                        )
                    else:
                        try:
                            if self.active_mimic is None:
                                raise RuntimeError("Mimic state has no active controller")
                            self.target_q = self.active_mimic.step(self.robot)
                            policy_steps += 1
                            self._send_position(
                                self.target_q,
                                self.active_mimic.kp_real,
                                self.active_mimic.kd_real,
                            )
                        except (FloatingPointError, RuntimeError, ValueError) as exc:
                            self._enter_damping(f"Mimic inference failed: {exc}")

                loop_steps += 1
                self._publish_api_status()
                now = time.perf_counter()
                elapsed = now - stats_start
                if self.state in (ControlState.HYBRID, ControlState.MIMIC) and elapsed >= 1.0:
                    print(
                        f"[Timing] mode={active_policy} low={loop_steps / elapsed:5.1f}Hz "
                        f"policy={policy_steps / elapsed:5.1f}Hz",
                        flush=True,
                    )
                    stats_start = now
                    loop_steps = 0
                    policy_steps = 0

                timer.waiting(step_start)

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected")
            self.logger.info("SHUTDOWN reason=keyboard_interrupt")
        finally:
            if self.api_server is not None:
                try:
                    self.api_server.stop()
                    print("[INFO] WebSocket API stopped")
                except Exception as exc:
                    print(f"[WARNING] Failed to stop WebSocket API cleanly: {exc}")
            # Preserve the pre-existing controller if the process is stopped
            # while it is still in read-only IDLE.  Once this FSM has claimed
            # control, always leave the robot in damping on shutdown.
            if self.control_claimed:
                try:
                    self.robot.get_robot_state()
                    self._send_damping()
                    print("[SAFETY] Final damping command sent")
                except Exception as exc:
                    print(f"[WARNING] Failed to send final damping command: {exc}")
            else:
                print("[INFO] Exited from read-only IDLE; motor commands were untouched")
            self.logger.info(
                "SHUTDOWN_COMPLETE final_state=%s control_claimed=%s",
                self.state.name,
                self.control_claimed,
            )
            print("Robot process finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified E1-21DOF Hybrid AMP and Mimic real-robot FSM"
    )
    parser.add_argument(
        "--hybrid-policy",
        type=Path,
        default=CURRENT_DIR / "policy/amp/walk_run.onnx",
    )
    parser.add_argument(
        "--mimic1-policy",
        type=Path,
        default=CURRENT_DIR / "policy/mimic/victory1_2.onnx",
    )
    parser.add_argument(
        "--mimic2-policy",
        type=Path,
        default=CURRENT_DIR / "policy/mimic/victory2_2.onnx",
    )
    parser.add_argument("--mimic1-motion-frames", type=int, default=None)
    parser.add_argument("--mimic2-motion-frames", type=int, default=None)
    parser.add_argument("--mimic1-start-frame", type=int, default=0)
    parser.add_argument("--mimic2-start-frame", type=int, default=0)
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--api-auth-token", default="droidpad")
    parser.add_argument("--api-status-hz", type=float, default=20.0)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=CURRENT_DIR / "logs/sim2real_fsm.log",
        help="Rotating diagnostic log file",
    )
    parser.add_argument("--log-max-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=5)
    parser.add_argument(
        "--no-api", action="store_true", help="Disable the WebSocket API server"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate all three policies without connecting to hardware",
    )
    return parser.parse_args()


def configure_file_logging(
    path: Path, max_bytes: int, backup_count: int
) -> Path:
    if max_bytes <= 0:
        raise ValueError("--log-max-bytes must be positive")
    if backup_count < 0:
        raise ValueError("--log-backup-count cannot be negative")
    path = path.expanduser()
    if not path.is_absolute():
        path = CURRENT_DIR / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s "
            "%(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("e1_fsm").info(
        "LOG_START path=%s max_bytes=%d backup_count=%d",
        path,
        max_bytes,
        backup_count,
    )
    return path


def main() -> None:
    args = parse_args()
    log_path = configure_file_logging(
        args.log_file, args.log_max_bytes, args.log_backup_count
    )
    print(f"[INFO] Diagnostic log: {log_path}")
    if args.validate_only:
        hybrid = HybridAmpInference(args.hybrid_policy.resolve())
        mimic1 = MimicInference(
            args.mimic1_policy.resolve(),
            args.mimic1_motion_frames,
            args.mimic1_start_frame,
            REAL_JOINT_NAMES,
        )
        mimic2 = MimicInference(
            args.mimic2_policy.resolve(),
            args.mimic2_motion_frames,
            args.mimic2_start_frame,
            REAL_JOINT_NAMES,
        )
        print(
            f"[VALID] hybrid_dt={hybrid.dt:.3f}s, mimic_dt={mimic1.dt:.3f}s, "
            f"victory1_frames={mimic1.policy.motion_frames}, "
            f"victory2_frames={mimic2.policy.motion_frames}"
        )
        logging.getLogger("e1_fsm").info("POLICY_VALIDATION success")
        return
    try:
        runner = UnifiedE1Fsm(
            args.hybrid_policy.resolve(),
            args.mimic1_policy.resolve(),
            args.mimic2_policy.resolve(),
            args.mimic1_motion_frames,
            args.mimic2_motion_frames,
            args.mimic1_start_frame,
            args.mimic2_start_frame,
            api_host=args.api_host,
            api_port=args.api_port,
            api_auth_token=args.api_auth_token,
            api_status_hz=args.api_status_hz,
            api_enabled=not args.no_api,
        )
        runner.run()
    except Exception:
        logging.getLogger("e1_fsm").exception("FATAL controller terminated")
        raise


if __name__ == "__main__":
    main()
