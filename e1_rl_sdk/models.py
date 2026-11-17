"""Public SDK request and response models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


@dataclass(frozen=True)
class RobotCommand:
    ctl_mode: str = "loco"
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_yaw: float = 0.0
    base_roll: float = 0.0
    base_pitch: float = 0.0
    base_yaw: float = 0.0
    base_z: float = 0.0
    loco_mode: str = "loco_walk_and_stand"
    arm_mode: str = "none"
    dance_mode: str = "none"
    fsm_action: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def stand(cls) -> "RobotCommand":
        return cls()

    @classmethod
    def walk(
        cls, vel_x: float = 0.0, vel_y: float = 0.0, vel_yaw: float = 0.0
    ) -> "RobotCommand":
        return cls(
            vel_x=_number(vel_x, "vel_x"),
            vel_y=_number(vel_y, "vel_y"),
            vel_yaw=_number(vel_yaw, "vel_yaw"),
        )

    @classmethod
    def damping(cls) -> "RobotCommand":
        return cls(loco_mode="loco_stop")

    @classmethod
    def release(cls) -> "RobotCommand":
        return cls(ctl_mode="none")

    @classmethod
    def dance(cls, number: int) -> "RobotCommand":
        if number not in (1, 2):
            raise ValueError("dance number must be 1 or 2")
        return cls(dance_mode=f"dance{number}")

    @classmethod
    def press_a(cls) -> "RobotCommand":
        return cls(ctl_mode="none", fsm_action="a")

    @classmethod
    def press_x(cls) -> "RobotCommand":
        return cls(ctl_mode="none", fsm_action="x")

    @classmethod
    def press_y(cls) -> "RobotCommand":
        return cls(ctl_mode="none", fsm_action="y")


@dataclass(frozen=True)
class CommandResult:
    """Server acknowledgement for one command.

    ``success`` means that the command passed the server-side state checks and
    was accepted by the controller.  A rejected command is a normal API result,
    not an SDK/protocol exception.
    """

    success: bool
    code: str
    message: str
    fsm_state: str = "unknown"
    active_policy: str = "none"
    dance_mode: str = "none"
    can_accept_velocity: bool = False
    can_accept_dance: bool = False
    can_accept_a: bool = False
    can_accept_x: bool = False
    can_accept_y: bool = True
    recovery_active: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommandResult":
        # Accept the original error envelope as well so a new SDK can still
        # report a rejection cleanly when connected to an older server.
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, Mapping):
                source = error
            else:
                source = {"code": "command_rejected", "message": str(error)}
            success = False
        else:
            source = payload
            success = bool(source.get("success", False))

        return cls(
            success=success,
            code=str(source.get("code", "ok" if success else "command_rejected")),
            message=str(
                source.get(
                    "message", "command accepted" if success else "command rejected"
                )
            ),
            fsm_state=str(source.get("fsm_state", "unknown")),
            active_policy=str(source.get("active_policy", "none")),
            dance_mode=str(source.get("dance_mode", "none")),
            can_accept_velocity=bool(source.get("can_accept_velocity", False)),
            can_accept_dance=bool(source.get("can_accept_dance", False)),
            can_accept_a=bool(source.get("can_accept_a", False)),
            can_accept_x=bool(source.get("can_accept_x", False)),
            can_accept_y=bool(source.get("can_accept_y", True)),
            recovery_active=bool(source.get("recovery_active", False)),
        )


@dataclass(frozen=True)
class RobotStatus:
    bus_voltage: float
    bus_current: float
    bus_energy: float
    vel_x: float
    vel_y: float
    vel_z: float
    joint_q: tuple[float, ...]
    joint_dq: tuple[float, ...]
    loco_mode: str
    arm_mode: str
    arm_action: str
    fsm_state: str = "unknown"
    active_policy: str = "none"
    dance_mode: str = "none"
    control_source: str = "gamepad"
    can_accept_velocity: bool = False
    can_accept_dance: bool = False
    can_accept_a: bool = False
    can_accept_x: bool = False
    can_accept_y: bool = True
    transitioning: bool = False
    recovery_active: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RobotStatus":
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, Mapping):
                message = str(error.get("message", error))
                if "fsm_state" in error:
                    message += (
                        f" [fsm_state={error.get('fsm_state')}, "
                        f"active_policy={error.get('active_policy')}, "
                        f"dance_mode={error.get('dance_mode')}]"
                    )
                raise ValueError(message)
            raise ValueError(str(error))

        def vector(name: str) -> tuple[float, ...]:
            values = payload.get(name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"{name} must be an array")
            result = tuple(_number(value, name) for value in values)
            if len(result) != 10:
                raise ValueError(f"{name} must contain 10 values")
            return result

        return cls(
            bus_voltage=_number(payload.get("bus_voltage"), "bus_voltage"),
            bus_current=_number(payload.get("bus_current"), "bus_current"),
            bus_energy=_number(payload.get("bus_energy"), "bus_energy"),
            vel_x=_number(payload.get("vel_x"), "vel_x"),
            vel_y=_number(payload.get("vel_y"), "vel_y"),
            vel_z=_number(payload.get("vel_z"), "vel_z"),
            joint_q=vector("joint_q"),
            joint_dq=vector("joint_dq"),
            loco_mode=str(payload.get("loco_mode")),
            arm_mode=str(payload.get("arm_mode", "none")),
            arm_action=str(payload.get("arm_action", "none")),
            fsm_state=str(payload.get("fsm_state", "unknown")),
            active_policy=str(payload.get("active_policy", "none")),
            dance_mode=str(payload.get("dance_mode", "none")),
            control_source=str(payload.get("control_source", "gamepad")),
            can_accept_velocity=bool(payload.get("can_accept_velocity", False)),
            can_accept_dance=bool(payload.get("can_accept_dance", False)),
            can_accept_a=bool(payload.get("can_accept_a", False)),
            can_accept_x=bool(payload.get("can_accept_x", False)),
            can_accept_y=bool(payload.get("can_accept_y", True)),
            transitioning=bool(payload.get("transitioning", False)),
            recovery_active=bool(payload.get("recovery_active", False)),
        )
