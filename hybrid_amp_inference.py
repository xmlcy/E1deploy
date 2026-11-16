"""Pure 21-DOF Hybrid AMP inference engine (no hardware loop or gamepad)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime


REAL_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
)

# The MJLab actor is exported in E1_21dof.xml order, which is also the
# hardware order used by the 13 leg/waist plus 8 arm gRPC interfaces.
ONNX_JOINT_NAMES = REAL_JOINT_NAMES


def _csv_strings(metadata: dict[str, str], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"Hybrid ONNX metadata is missing {key!r}")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _metadata_joint_values(
    metadata: dict[str, str], key: str, size: int
) -> np.ndarray:
    try:
        values = np.asarray(
            [float(item) for item in _csv_strings(metadata, key)], dtype=np.float32
        )
    except ValueError as exc:
        raise ValueError(f"Invalid floating-point Hybrid metadata in {key!r}") from exc
    if values.shape == (1,):
        values = np.full(size, values.item(), dtype=np.float32)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Hybrid metadata {key!r} has shape {values.shape}; expected ({size},)"
        )
    return values


def _euler_xyz_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in euler)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )


def _quat_rotate_inverse(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = float(quat_wxyz[0])
    xyz = np.asarray(quat_wxyz[1:4], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return (
        vector * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    )


class HybridAmpInference:
    """Stateful Hybrid AMP actor matching the original 50-Hz deployment."""

    dt = 0.02
    decimation = 1
    policy_dt = dt * decimation
    num_actions = 21
    obs_per_step = 78
    history_length = 5
    clip_observations = 100.0
    clip_actions = 100.0
    scale_ang_vel = 0.2
    scale_joint_vel = 0.05

    max_forward_vel = 1.5
    max_backward_vel = 0.6
    max_lateral_vel = 0.5
    max_yaw_vel = 1.2
    walk_gait_cycle = 1.1667
    run_gait_cycle = 0.78
    gait_phase_offset = np.asarray((0.38, 0.88), dtype=np.float32)
    walk_air_ratio = np.asarray((0.38, 0.38), dtype=np.float32)
    run_air_ratio = np.asarray((0.60, 0.60), dtype=np.float32)
    gait_command_threshold = 0.1
    run_start_forward_speed = 0.80
    run_full_forward_speed = 0.95
    run_lateral_start = 0.10
    run_lateral_full_walk = 0.25

    def __init__(self, policy_path: Path) -> None:
        if not policy_path.is_file():
            raise FileNotFoundError(f"Hybrid ONNX model not found: {policy_path}")
        print(f"[INFO] Loading Hybrid AMP ONNX: {policy_path}")
        self.session = onnxruntime.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Hybrid policy must have one input/output, got {len(inputs)}/{len(outputs)}"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        input_width = inputs[0].shape[-1]
        expected_width = self.obs_per_step * self.history_length
        if isinstance(input_width, int) and input_width != expected_width:
            raise ValueError(
                f"Hybrid policy expects {input_width} observations; expected {expected_width}"
            )
        output_width = outputs[0].shape[-1]
        if isinstance(output_width, int) and output_width != self.num_actions:
            raise ValueError(
                f"Hybrid policy returns {output_width} actions; expected {self.num_actions}"
            )

        metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        supported_task_types = {
            "velocity_commanded_hybrid_amp",
            "velocity_commanded_hybrid_rough_amp",
        }
        if metadata.get("task_type") not in supported_task_types:
            raise ValueError(
                "ONNX is not a velocity-commanded Hybrid AMP policy; "
                f"got task_type={metadata.get('task_type')!r}"
            )
        if metadata.get("actor_history_length") != str(self.history_length):
            raise ValueError("Hybrid actor history length does not match deployment")
        if metadata.get("actor_frame_dim") != str(self.obs_per_step):
            raise ValueError("Hybrid actor frame dimension does not match deployment")
        metadata_joint_names = _csv_strings(metadata, "joint_names")
        if metadata_joint_names != ONNX_JOINT_NAMES:
            raise ValueError(
                "Hybrid ONNX joint order does not match E1_21dof.xml/hardware order:\n"
                f"ONNX: {metadata_joint_names}\nExpected: {ONNX_JOINT_NAMES}"
            )
        self.default_pos_onnx = _metadata_joint_values(
            metadata, "default_joint_pos", self.num_actions
        )
        self.kp_onnx = _metadata_joint_values(
            metadata, "joint_stiffness", self.num_actions
        )
        self.kd_onnx = _metadata_joint_values(
            metadata, "joint_damping", self.num_actions
        )
        self.action_scale = _metadata_joint_values(
            metadata, "action_scale", self.num_actions
        )

        self.idx_to_onnx = np.asarray(
            [REAL_JOINT_NAMES.index(name) for name in ONNX_JOINT_NAMES], dtype=np.int32
        )
        self.idx_to_real = np.asarray(
            [ONNX_JOINT_NAMES.index(name) for name in REAL_JOINT_NAMES], dtype=np.int32
        )
        self.default_pos_real = self.default_pos_onnx[self.idx_to_real]
        self.kp_real = self.kp_onnx[self.idx_to_real]
        self.kd_real = self.kd_onnx[self.idx_to_real]
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.command_vel = np.zeros(3, dtype=np.float32)
        self.obs_history = np.zeros(expected_width, dtype=np.float32)
        self.gait_phase_progress = 0.0
        self.gait_phase = self.gait_phase_offset.copy()
        self.run_ratio = 0.0
        self.gait_cycle = self.walk_gait_cycle
        self.gait_air_ratio = self.walk_air_ratio.copy()
        print("[SUCCESS] Hybrid AMP policy validated: obs=390, action=21")

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _is_moving(self) -> bool:
        return bool(
            np.linalg.norm(self.command_vel[:2]) + abs(float(self.command_vel[2]))
            > self.gait_command_threshold
        )

    def is_moving(self) -> bool:
        """Whether the dead-zoned, scaled velocity command is non-zero."""
        return self._is_moving()

    def _refresh_gait_parameters(self) -> None:
        forward = self._smoothstep(
            (float(self.command_vel[0]) - self.run_start_forward_speed)
            / (self.run_full_forward_speed - self.run_start_forward_speed)
        )
        lateral_walk = self._smoothstep(
            (abs(float(self.command_vel[1])) - self.run_lateral_start)
            / (self.run_lateral_full_walk - self.run_lateral_start)
        )
        self.run_ratio = forward * (1.0 - lateral_walk)
        self.gait_cycle = self.walk_gait_cycle + self.run_ratio * (
            self.run_gait_cycle - self.walk_gait_cycle
        )
        self.gait_air_ratio = (
            self.walk_air_ratio
            + self.run_ratio * (self.run_air_ratio - self.walk_air_ratio)
        ).astype(np.float32)

    def _update_phase(self) -> None:
        self._refresh_gait_parameters()
        if self._is_moving():
            self.gait_phase_progress += self.policy_dt / self.gait_cycle
        self.gait_phase = np.mod(
            self.gait_phase_progress + self.gait_phase_offset, 1.0
        ).astype(np.float32)

    def set_gamepad_command(self, raw_x: float, raw_y: float, raw_yaw: float) -> None:
        x_scale = self.max_forward_vel if raw_x >= 0.0 else self.max_backward_vel
        self.command_vel[0] = raw_x * x_scale
        self.command_vel[1] = np.clip(raw_y, -self.max_lateral_vel, self.max_lateral_vel)
        self.command_vel[2] = raw_yaw * self.max_yaw_vel
        self._refresh_gait_parameters()

    def set_velocity_command(self, vel_x: float, vel_y: float, vel_yaw: float) -> None:
        """Set a physical-unit command received from the public API."""
        self.command_vel[0] = np.clip(
            vel_x, -self.max_backward_vel, self.max_forward_vel
        )
        self.command_vel[1] = np.clip(
            vel_y, -self.max_lateral_vel, self.max_lateral_vel
        )
        self.command_vel[2] = np.clip(vel_yaw, -self.max_yaw_vel, self.max_yaw_vel)
        self._refresh_gait_parameters()

    def _current_observation(self, robot) -> np.ndarray:
        q_real = np.asarray(
            list(robot.legState.position[:13]) + list(robot.armState.position[:8]),
            dtype=np.float32,
        )
        dq_real = np.asarray(
            list(robot.legState.velocity[:13]) + list(robot.armState.velocity[:8]),
            dtype=np.float32,
        )
        q_onnx = q_real[self.idx_to_onnx]
        dq_onnx = dq_real[self.idx_to_onnx]
        euler = np.asarray(robot.legState.imu_euler[:3], dtype=np.float64)
        euler[euler > math.pi] -= 2.0 * math.pi
        gravity = _quat_rotate_inverse(
            _euler_xyz_to_quat_wxyz(euler), np.asarray((0.0, 0.0, -1.0))
        ).astype(np.float32)
        ang_vel = np.asarray(robot.legState.imu_gyro[:3], dtype=np.float32) * self.scale_ang_vel
        moving = np.float32(self._is_moving())
        obs = np.concatenate(
            (
                ang_vel,
                gravity,
                self.command_vel,
                q_onnx - self.default_pos_onnx,
                dq_onnx * self.scale_joint_vel,
                self.action,
                np.sin(2.0 * np.pi * self.gait_phase) * moving,
                np.cos(2.0 * np.pi * self.gait_phase) * moving,
                self.gait_air_ratio,
            )
        ).astype(np.float32)
        if obs.shape != (self.obs_per_step,) or not np.all(np.isfinite(obs)):
            raise FloatingPointError(f"Invalid Hybrid observation: shape={obs.shape}")
        return np.clip(obs, -self.clip_observations, self.clip_observations)

    def _push_observation(self, robot) -> np.ndarray:
        obs = self._current_observation(robot)
        self.obs_history = np.roll(self.obs_history, -self.obs_per_step)
        self.obs_history[-self.obs_per_step :] = obs
        return self.obs_history.reshape(1, -1)

    def reset(self, robot) -> None:
        self.command_vel.fill(0.0)
        self.gait_phase_progress = 0.0
        self.gait_phase = self.gait_phase_offset.copy()
        self.run_ratio = 0.0
        self.gait_cycle = self.walk_gait_cycle
        self.gait_air_ratio = self.walk_air_ratio.copy()
        self.action.fill(0.0)
        initial_frame = self._current_observation(robot)
        self.obs_history = np.tile(initial_frame, self.history_length).astype(np.float32)

    def _infer(self, robot) -> np.ndarray:
        action = self.session.run(
            [self.output_name], {self.input_name: self._push_observation(robot)}
        )[0].reshape(-1)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Hybrid policy returned non-finite actions")
        self.action[:] = np.clip(action, -self.clip_actions, self.clip_actions)
        target_onnx = self.default_pos_onnx + self.action * self.action_scale
        target = target_onnx[self.idx_to_real]
        self._update_phase()
        return target.astype(np.float32)

    def step(self, robot, raw_command: tuple[float, float, float]) -> np.ndarray:
        self.set_gamepad_command(*raw_command)
        return self._infer(robot)

    def step_velocity(
        self, robot, velocity_command: tuple[float, float, float]
    ) -> np.ndarray:
        """Run one policy step from an API command in m/s and rad/s."""
        self.set_velocity_command(*velocity_command)
        return self._infer(robot)
