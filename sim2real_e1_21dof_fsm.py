"""Unified 50-Hz E1-21DOF Hybrid AMP / Mimic real-robot FSM.

IDLE is deliberately read-only: launching this process does not overwrite the
controller that was active before it. X claims control, A starts Hybrid, and
only a stationary Hybrid state may enter either Mimic policy.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import from this directory deliberately. Importing ``fsm.*`` here would load
# the obsolete IsaacLab deployment under ``sim2real/fsm`` instead.
from gamepad_controller import GamepadController  # noqa: E402
from hybrid_amp_inference import (  # noqa: E402
    REAL_JOINT_NAMES,
    HybridAmpInference,
)
from mimic_inference import MimicInference  # noqa: E402


class ControlState(Enum):
    IDLE = auto()
    GO_TO_DEFAULT = auto()
    STANDBY = auto()
    GO_TO_MIMIC_START = auto()
    GO_TO_HYBRID_START = auto()
    HYBRID = auto()
    MIMIC = auto()
    DAMPING = auto()
    RECOVERY_HOLD = auto()


class UnifiedE1Fsm:
    dt = 0.02
    # A long open-loop pose interpolation cannot actively reject disturbances.
    # Keep mode transitions short, then let the selected policy take over.
    interpolation_duration = 0.2
    fall_tilt_threshold_deg = 50.0
    fall_hard_tilt_threshold_deg = 75.0
    fall_detection_frames = 3
    recovery_upright_tilt_deg = 30.0
    recovery_gain_ramp_duration = 0.75
    recovery_kp_fraction = 0.35
    leg_damping = 4.0
    arm_damping = 2.0

    def __init__(
        self,
        hybrid_policy: Path,
        mimic1_policy: Path,
        mimic2_policy: Path,
        mimic1_motion_frames: int | None,
        mimic2_motion_frames: int | None,
        mimic1_start_frame: int,
        mimic2_start_frame: int,
    ) -> None:
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
        self.recovery_target = np.zeros(21, dtype=np.float32)
        self.recovery_elapsed = 0.0
        self.fall_counter = 0

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

    def _enter_damping(self, reason: str) -> None:
        self.state = ControlState.DAMPING
        self.control_claimed = True
        self.fall_counter = 0
        self.hybrid.command_vel.fill(0.0)
        self._send_damping()
        print(f"\n[SAFETY] {reason}: entered DAMPING")
        print("[SAFETY] X: stiffen in-place for assisted lifting; B: exit")

    def _check_fall(self) -> None:
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
            self._enter_damping(text)

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
            self.state = ControlState.HYBRID
            print("[SUCCESS] Hybrid default pose reached: walk-run inference started")

    def _start_recovery_hold(self) -> None:
        self.recovery_target = self._joint_positions()
        self.recovery_elapsed = 0.0
        self.state = ControlState.RECOVERY_HOLD
        print("[SAFETY] RECOVERY_HOLD: ramping stiffness at the current pose")
        print("[SAFETY] Lift/support upright, then press X again for default pose")

    def _run_recovery_hold(self) -> None:
        ratio = min(self.recovery_elapsed / self.recovery_gain_ramp_duration, 1.0)
        ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        kp = self.hybrid.kp_real * (self.recovery_kp_fraction * ratio)
        baseline_kd = np.asarray(
            [self.leg_damping] * 13 + [self.arm_damping] * 8,
            dtype=np.float32,
        )
        kd = baseline_kd + ratio * (self.hybrid.kd_real - baseline_kd)
        self._send_position(self.recovery_target, kp, kd)
        self.recovery_elapsed += self.dt

    def run(self) -> None:
        print("\n" + "=" * 66)
        print("E1-21DOF Unified Sim2Real FSM (50 Hz)")
        print("  startup : IDLE/read-only; no damping command is sent")
        print("  [X]     : default pose / assisted recovery")
        print("  [A]     : STANDBY -> Hybrid; Mimic -> Hybrid")
        print("  [START] : stationary Hybrid -> victory1 (button 11)")
        print("  [BACK]  : stationary Hybrid -> victory2 (button 10)")
        print("  Mimic1/Mimic2 cannot switch directly; press A first")
        print("  [Y]     : damping")
        print("  [B]     : damping and exit")
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

                if self.gamepad.get_button_b():
                    print("\n[EMERGENCY] B pressed: damping and exit")
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
                        self._start_recovery_hold()
                    else:
                        self._send_damping()

                elif self.state == ControlState.RECOVERY_HOLD:
                    if pressed["x"]:
                        tilt = self._torso_tilt_deg()
                        if (
                            self.recovery_elapsed >= self.recovery_gain_ramp_duration
                            and math.isfinite(tilt)
                            and tilt <= self.recovery_upright_tilt_deg
                        ):
                            print(f"[SAFETY] Upright at {tilt:.1f} deg: going to default pose")
                            self._begin_default_transition()
                        else:
                            print(
                                f"[SAFETY] X ignored: recovery ramp/tilt not ready "
                                f"(tilt={tilt:.1f} deg)"
                            )
                    if self.state == ControlState.RECOVERY_HOLD:
                        self._run_recovery_hold()

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
                        self.hybrid.reset(self.robot)
                        self.target_q = self.hybrid.default_pos_real.copy()
                        self.state = ControlState.HYBRID
                        active_policy = "hybrid"
                        stats_start = time.perf_counter()
                        loop_steps = 0
                        policy_steps = 0
                        print("[SUCCESS] Hybrid walk-run inference started")
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
                    raw_command = self.gamepad.get_commands()
                    self.hybrid.set_gamepad_command(*raw_command)
                    if pressed["start"] and pressed["back"]:
                        print("[INFO] START+BACK ignored: select only one Mimic policy")
                    elif pressed["start"] or pressed["back"]:
                        if self.hybrid.is_moving():
                            print(
                                "[INFO] Mimic request ignored: release the sticks and "
                                "wait for a zero velocity command before Mimic"
                            )
                        else:
                            mimic_name = "victory1" if pressed["start"] else "victory2"
                            button_name = "START" if pressed["start"] else "BACK"
                            print(
                                f"[INFO] {button_name} pressed in stationary Hybrid: "
                                f"moving to {mimic_name} start pose"
                            )
                            self._begin_mimic_transition(mimic_name)
                    elif pressed["x"]:
                        print("[INFO] Leaving Hybrid: returning to default pose")
                        self.hybrid.command_vel.fill(0.0)
                        self._begin_default_transition()
                    else:
                        try:
                            self.target_q = self.hybrid.step(
                                self.robot, raw_command
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
                    if pressed["a"]:
                        print(
                            f"[INFO] Leaving {self.active_mimic_name}: "
                            "moving to Hybrid default pose"
                        )
                        self._begin_transition(
                            ControlState.GO_TO_HYBRID_START,
                            self.hybrid.default_pos_real,
                            self.hybrid.kp_real,
                            self.hybrid.kd_real,
                        )
                    elif pressed["x"]:
                        print("[INFO] Leaving Mimic: returning to default pose")
                        self._begin_default_transition()
                    elif pressed["start"] or pressed["back"]:
                        print(
                            "[INFO] Direct Mimic switching is disabled: "
                            "press A to return to Hybrid first"
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
        finally:
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate all three policies without connecting to hardware",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        return
    runner = UnifiedE1Fsm(
        args.hybrid_policy.resolve(),
        args.mimic1_policy.resolve(),
        args.mimic2_policy.resolve(),
        args.mimic1_motion_frames,
        args.mimic2_motion_frames,
        args.mimic1_start_frame,
        args.mimic2_start_frame,
    )
    runner.run()


if __name__ == "__main__":
    main()
