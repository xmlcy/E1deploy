"""Wire-level models and validation for the documented RL WebSocket API."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


AUTH_REQUEST = "droidpad"
AUTH_RESPONSE = "droidup"

CTL_MODES = frozenset(("none", "loco"))
COMMAND_LOCO_MODES = frozenset(("loco_stop", "loco_walk_and_stand"))
STATUS_LOCO_MODES = frozenset(("none", "loco_stop", "loco_walk_and_stand", "loco_fall"))
DANCE_MODES = frozenset(("none", "dance1", "dance2"))


class ProtocolError(ValueError):
    """Raised when a WebSocket payload doesn't conform to the API schema."""


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ProtocolError(f"{field} must be a finite number")
    return number


def _string_choice(value: Any, field: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        options = ", ".join(sorted(choices))
        raise ProtocolError(f"{field} must be one of: {options}")
    return value


def finite_or_zero(value: Any) -> float:
    """Convert a hardware value to JSON-safe float, substituting zero if invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


@dataclass(frozen=True)
class RobotCommand:
    """Validated ``robot_cmd`` request.

    ``dance_mode`` is a backwards-compatible extension because the supplied
    document says dances are WebSocket-triggerable but doesn't assign a field.
    Omitting it is identical to sending ``"none"``.
    """

    ctl_mode: str = "none"
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

    @classmethod
    def from_payload(cls, payload: Any) -> "RobotCommand":
        if not isinstance(payload, Mapping):
            raise ProtocolError("robot_cmd must be a JSON object")

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ProtocolError(f"unknown robot_cmd fields: {', '.join(unknown)}")

        ctl_mode = _string_choice(payload.get("ctl_mode", "none"), "ctl_mode", CTL_MODES)
        loco_mode = _string_choice(
            payload.get("loco_mode", "loco_walk_and_stand"),
            "loco_mode",
            COMMAND_LOCO_MODES,
        )
        arm_mode = payload.get("arm_mode", "none")
        if arm_mode != "none":
            raise ProtocolError('arm_mode is reserved and must be "none"')

        dance_mode = _string_choice(
            payload.get("dance_mode", "none"), "dance_mode", DANCE_MODES
        )
        velocities = {
            name: _finite_float(payload.get(name, 0.0), name)
            for name in ("vel_x", "vel_y", "vel_yaw")
        }
        if dance_mode != "none":
            if ctl_mode != "loco" or loco_mode != "loco_walk_and_stand":
                raise ProtocolError(
                    "dance_mode requires ctl_mode=loco and "
                    "loco_mode=loco_walk_and_stand"
                )
            if any(abs(value) > 1.0e-9 for value in velocities.values()):
                raise ProtocolError("dance_mode requires zero velocity")

        return cls(
            ctl_mode=ctl_mode,
            **velocities,
            base_roll=_finite_float(payload.get("base_roll", 0.0), "base_roll"),
            base_pitch=_finite_float(payload.get("base_pitch", 0.0), "base_pitch"),
            base_yaw=_finite_float(payload.get("base_yaw", 0.0), "base_yaw"),
            base_z=_finite_float(payload.get("base_z", 0.0), "base_z"),
            loco_mode=loco_mode,
            arm_mode="none",
            dance_mode=dance_mode,
        )

    def to_dict(self, *, include_extensions: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_extensions and self.dance_mode == "none":
            result.pop("dance_mode")
        return result

    @classmethod
    def safe_stand(cls) -> "RobotCommand":
        return cls(ctl_mode="loco", loco_mode="loco_walk_and_stand")


@dataclass(frozen=True)
class RobotStatus:
    """Documented ``robot_status`` response."""

    bus_voltage: float = 0.0
    bus_current: float = 0.0
    bus_energy: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    joint_q: tuple[float, ...] = (0.0,) * 10
    joint_dq: tuple[float, ...] = (0.0,) * 10
    loco_mode: str = "none"
    arm_mode: str = "none"
    arm_action: str = "none"
    fsm_state: str = "unknown"
    active_policy: str = "none"
    dance_mode: str = "none"
    control_source: str = "gamepad"
    can_accept_velocity: bool = False
    can_accept_dance: bool = False
    transitioning: bool = False
    recovery_active: bool = False

    def __post_init__(self) -> None:
        if len(self.joint_q) != 10 or len(self.joint_dq) != 10:
            raise ProtocolError("joint_q and joint_dq must each contain 10 values")
        if self.loco_mode not in STATUS_LOCO_MODES:
            raise ProtocolError(f"invalid status loco_mode: {self.loco_mode}")

    @classmethod
    def from_payload(cls, payload: Any) -> "RobotStatus":
        if not isinstance(payload, Mapping):
            raise ProtocolError("robot_status must be a JSON object")
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
            else:
                message = str(error)
            raise ProtocolError(f"server rejected request: {message}")

        def vector(name: str) -> tuple[float, ...]:
            value = payload.get(name)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ProtocolError(f"{name} must be an array of 10 numbers")
            result = tuple(_finite_float(item, name) for item in value)
            if len(result) != 10:
                raise ProtocolError(f"{name} must contain 10 numbers")
            return result

        return cls(
            bus_voltage=_finite_float(payload.get("bus_voltage"), "bus_voltage"),
            bus_current=_finite_float(payload.get("bus_current"), "bus_current"),
            bus_energy=_finite_float(payload.get("bus_energy"), "bus_energy"),
            vel_x=_finite_float(payload.get("vel_x"), "vel_x"),
            vel_y=_finite_float(payload.get("vel_y"), "vel_y"),
            vel_z=_finite_float(payload.get("vel_z"), "vel_z"),
            joint_q=vector("joint_q"),
            joint_dq=vector("joint_dq"),
            loco_mode=_string_choice(
                payload.get("loco_mode"), "loco_mode", STATUS_LOCO_MODES
            ),
            arm_mode=str(payload.get("arm_mode", "none")),
            arm_action=str(payload.get("arm_action", "none")),
            fsm_state=str(payload.get("fsm_state", "unknown")),
            active_policy=str(payload.get("active_policy", "none")),
            dance_mode=str(payload.get("dance_mode", "none")),
            control_source=str(payload.get("control_source", "gamepad")),
            can_accept_velocity=bool(payload.get("can_accept_velocity", False)),
            can_accept_dance=bool(payload.get("can_accept_dance", False)),
            transitioning=bool(payload.get("transitioning", False)),
            recovery_active=bool(payload.get("recovery_active", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["joint_q"] = list(self.joint_q)
        result["joint_dq"] = list(self.joint_dq)
        return result
