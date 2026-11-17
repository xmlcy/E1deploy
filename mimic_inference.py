"""Pure 21-DOF Mimic inference engine (no hardware loop or gamepad)."""

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
EXPECTED_OBSERVATIONS = (
    "command", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"
)
OBSERVATION_SIZE = 111
ACTION_SIZE = 21


def _csv_strings(metadata: dict[str, str], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"ONNX metadata is missing {key!r}")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_floats(metadata: dict[str, str], key: str) -> np.ndarray:
    try:
        return np.asarray(
            [float(item) for item in _csv_strings(metadata, key)], dtype=np.float32
        )
    except ValueError as exc:
        raise ValueError(f"Invalid floating-point ONNX metadata in {key!r}") from exc


def _effort_limit(joint_name: str) -> float:
    if joint_name == "waist_yaw_joint":
        return 60.0
    limits = (
        ("_hip_pitch_joint", 120.0),
        ("_hip_roll_joint", 60.0),
        ("_hip_yaw_joint", 36.0),
        ("_knee_joint", 120.0),
        ("_ankle_pitch_joint", 36.0),
        ("_ankle_roll_joint", 30.0),
        ("_shoulder_pitch_joint", 60.0),
        ("_shoulder_roll_joint", 36.0),
        ("_shoulder_yaw_joint", 15.0),
        ("_elbow_joint", 60.0),
    )
    for suffix, limit in limits:
        if joint_name.endswith(suffix):
            return limit
    raise ValueError(f"No E1 effort limit is defined for {joint_name!r}")


def _infer_motion_frames(policy_path: Path) -> int | None:
    try:
        import onnx
    except ImportError:
        return None
    model = onnx.load(str(policy_path), load_external_data=False)
    for initializer in model.graph.initializer:
        if (
            initializer.name.startswith("joint_pos")
            and len(initializer.dims) == 2
            and int(initializer.dims[1]) == ACTION_SIZE
        ):
            return int(initializer.dims[0])
    return None


class MimicPolicy:
    """Validated ONNX actor and its bundled reference-motion stream."""

    def __init__(self, policy_path: Path, motion_frames: int | None) -> None:
        if not policy_path.is_file():
            raise FileNotFoundError(f"Mimic ONNX model not found: {policy_path}")
        self.path = policy_path.resolve()
        print(f"[INFO] Loading Mimic ONNX: {policy_path}")
        self.session = onnxruntime.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        inputs = {item.name: tuple(item.shape) for item in self.session.get_inputs()}
        outputs = {item.name: tuple(item.shape) for item in self.session.get_outputs()}
        if inputs.get("obs") != (1, OBSERVATION_SIZE):
            raise ValueError(f"Expected Mimic obs (1,111), got {inputs.get('obs')}")
        if inputs.get("time_step") != (1, 1):
            raise ValueError(
                f"Expected Mimic time_step (1,1), got {inputs.get('time_step')}"
            )
        for name, shape in {
            "actions": (1, ACTION_SIZE),
            "joint_pos": (1, ACTION_SIZE),
            "joint_vel": (1, ACTION_SIZE),
        }.items():
            if outputs.get(name) != shape:
                raise ValueError(f"Expected Mimic {name} {shape}, got {outputs.get(name)}")

        if _csv_strings(self.metadata, "observation_names") != EXPECTED_OBSERVATIONS:
            raise ValueError("Mimic observation_names do not match the 111-D deployment")
        if self.metadata.get("anchor_body_name") != "torso_link":
            raise ValueError("Mimic anchor_body_name must be 'torso_link'")
        self.joint_names = _csv_strings(self.metadata, "joint_names")
        if len(self.joint_names) != ACTION_SIZE or set(self.joint_names) != set(
            REAL_JOINT_NAMES
        ):
            raise ValueError("Mimic joint_names do not match the E1 21-DOF joint set")

        self.default_joint_pos = _csv_floats(self.metadata, "default_joint_pos")
        self.kp = _csv_floats(self.metadata, "joint_stiffness")
        self.kd = _csv_floats(self.metadata, "joint_damping")
        metadata_scale = _csv_floats(self.metadata, "action_scale")
        for name, value in (
            ("default_joint_pos", self.default_joint_pos),
            ("joint_stiffness", self.kp),
            ("joint_damping", self.kd),
            ("action_scale", metadata_scale),
        ):
            if value.shape != (ACTION_SIZE,):
                raise ValueError(f"Mimic {name} has shape {value.shape}, expected (21,)")
        effort = np.asarray(
            [_effort_limit(name) for name in self.joint_names], dtype=np.float32
        )
        self.action_scale = 0.25 * effort / self.kp
        if not np.allclose(self.action_scale, metadata_scale, atol=6.0e-4, rtol=0.0):
            raise ValueError("Mimic action scale disagrees with ONNX metadata")

        inferred_frames = _infer_motion_frames(policy_path)
        if motion_frames is None:
            if inferred_frames is None:
                raise ValueError("Install onnx or pass --motion-frames explicitly")
            motion_frames = inferred_frames
        elif inferred_frames is not None and inferred_frames != motion_frames:
            raise ValueError(
                f"--motion-frames={motion_frames} disagrees with ONNX length {inferred_frames}"
            )
        if motion_frames <= 1:
            raise ValueError("Mimic motion must contain at least two frames")
        self.motion_frames = motion_frames
        self.zero_observation = np.zeros((1, OBSERVATION_SIZE), dtype=np.float32)
        print(
            f"[SUCCESS] Mimic policy validated: obs=111, action=21, "
            f"motion_frames={self.motion_frames}"
        )

    @staticmethod
    def _time_step(frame: int) -> np.ndarray:
        return np.asarray([[frame]], dtype=np.float32)

    def reference(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        joint_pos, joint_vel = self.session.run(
            ["joint_pos", "joint_vel"],
            {
                "obs": self.zero_observation,
                "time_step": self._time_step(frame % self.motion_frames),
            },
        )
        return (
            np.asarray(joint_pos, dtype=np.float32).reshape(ACTION_SIZE),
            np.asarray(joint_vel, dtype=np.float32).reshape(ACTION_SIZE),
        )

    def infer_and_get_next_reference(
        self, observation: np.ndarray, next_frame: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action, joint_pos, joint_vel = self.session.run(
            ["actions", "joint_pos", "joint_vel"],
            {
                "obs": observation.reshape(1, OBSERVATION_SIZE).astype(np.float32),
                "time_step": self._time_step(next_frame % self.motion_frames),
            },
        )
        return (
            np.asarray(action, dtype=np.float32).reshape(ACTION_SIZE),
            np.asarray(joint_pos, dtype=np.float32).reshape(ACTION_SIZE),
            np.asarray(joint_vel, dtype=np.float32).reshape(ACTION_SIZE),
        )


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


def _quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = float(quat[0])
    xyz = np.asarray(quat[1:4], dtype=np.float64)
    return (
        vector * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    )


class MimicInference:
    dt = 0.02
    decimation = 1

    def __init__(
        self,
        policy_path: Path,
        motion_frames: int | None,
        motion_start_frame: int,
        hardware_joint_names: tuple[str, ...],
    ) -> None:
        self.policy = MimicPolicy(policy_path, motion_frames)
        self.motion_start_frame = motion_start_frame % self.policy.motion_frames
        self.idx_policy_to_real = np.asarray(
            [hardware_joint_names.index(name) for name in self.policy.joint_names],
            dtype=np.int32,
        )
        self.idx_real_to_policy = np.asarray(
            [self.policy.joint_names.index(name) for name in hardware_joint_names],
            dtype=np.int32,
        )
        start_policy, _ = self.policy.reference(self.motion_start_frame)
        self.start_pose_real = start_policy[self.idx_real_to_policy]
        self.kp_real = self.policy.kp[self.idx_real_to_policy]
        self.kd_real = self.policy.kd[self.idx_real_to_policy]
        self.action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.reference_frame = 0
        self.reference_joint_pos = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.reference_joint_vel = np.zeros(ACTION_SIZE, dtype=np.float32)

    @staticmethod
    def _joint_state(robot) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(
            list(robot.legState.position[:13]) + list(robot.armState.position[:8]),
            dtype=np.float32,
        )
        dq = np.asarray(
            list(robot.legState.velocity[:13]) + list(robot.armState.velocity[:8]),
            dtype=np.float32,
        )
        return q, dq

    def reset(self) -> None:
        self.action.fill(0.0)
        self.reference_frame = (self.motion_start_frame + 1) % self.policy.motion_frames
        self.reference_joint_pos, self.reference_joint_vel = self.policy.reference(
            self.reference_frame
        )

    def _observation(self, robot) -> np.ndarray:
        q_real, dq_real = self._joint_state(robot)
        q_policy = q_real[self.idx_policy_to_real]
        dq_policy = dq_real[self.idx_policy_to_real]
        euler = np.asarray(robot.legState.imu_euler[:3], dtype=np.float64)
        if not np.all(np.isfinite(euler)):
            raise FloatingPointError("Invalid torso IMU orientation")
        euler[euler > math.pi] -= 2.0 * math.pi
        gravity = _quat_rotate_inverse(
            _euler_xyz_to_quat_wxyz(euler), np.asarray((0.0, 0.0, -1.0))
        ).astype(np.float32)
        gyro = np.asarray(robot.legState.imu_gyro[:3], dtype=np.float32)
        obs = np.concatenate(
            (
                self.reference_joint_pos,
                self.reference_joint_vel,
                gravity,
                gyro,
                q_policy - self.policy.default_joint_pos,
                dq_policy,
                self.action,
            )
        ).astype(np.float32)
        if obs.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(obs)):
            raise FloatingPointError(f"Invalid Mimic observation: shape={obs.shape}")
        return obs.reshape(1, -1)

    def step(self, robot) -> np.ndarray:
        next_frame = (self.reference_frame + 1) % self.policy.motion_frames
        action, next_pos, next_vel = self.policy.infer_and_get_next_reference(
            self._observation(robot), next_frame
        )
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Mimic policy returned non-finite actions")
        self.action[:] = np.clip(action, -100.0, 100.0)
        target_policy = (
            self.policy.default_joint_pos + self.action * self.policy.action_scale
        )
        target_real = target_policy[self.idx_real_to_policy]
        self.reference_frame = next_frame
        self.reference_joint_pos = next_pos
        self.reference_joint_vel = next_vel
        if not np.all(np.isfinite(target_real)):
            raise FloatingPointError("Mimic target contains non-finite values")
        return target_real.astype(np.float32)
