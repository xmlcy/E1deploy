"""Threaded WebSocket server bridging network clients to the 50 Hz FSM."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from contextlib import suppress
from itertools import count
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .protocol import (
    AUTH_REQUEST,
    AUTH_RESPONSE,
    ProtocolError,
    RobotCommand,
    RobotStatus,
)


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class RlWebSocketServer:
    """A non-blocking, thread-safe WebSocket facade for the controller.

    Network I/O runs in a daemon thread. The real-time-ish control loop only
    performs short lock-protected snapshot reads and writes.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        *,
        auth_token: str = AUTH_REQUEST,
        status_hz: float = 20.0,
        auth_timeout: float = 5.0,
        heartbeat_timeout: float = 2.5,
    ) -> None:
        if not 1.0 <= status_hz <= 50.0:
            raise ValueError("status_hz must be between 1 and 50")
        if heartbeat_timeout < 0.5:
            raise ValueError("heartbeat_timeout must be at least 0.5 seconds")
        self.host = host
        self.port = int(port)
        self.bound_port = int(port)
        self.auth_token = auth_token
        self.status_hz = float(status_hz)
        self.auth_timeout = float(auth_timeout)
        self.heartbeat_timeout = float(heartbeat_timeout)

        self._lock = threading.Lock()
        self._command = RobotCommand()
        self._command_generation = 0
        self._owner: int | None = None
        self._dance_lease_owner: int | None = None
        self._status = RobotStatus()
        self._dance_requests: deque[str] = deque(maxlen=8)
        self._fsm_requests: deque[str] = deque(maxlen=8)
        self._client_dance_modes: dict[int, str] = {}
        self._client_ids = count(1)

        self._started = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    def start(self, timeout: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._started.clear()
        self._stop_requested.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="rl-websocket-api",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout):
            raise TimeoutError("timed out starting RL WebSocket API")
        if self._startup_error is not None:
            raise RuntimeError("failed to start RL WebSocket API") from self._startup_error
        LOGGER.info("WS_SERVER_STARTED host=%s port=%d", self.host, self.bound_port)

    def stop(self, timeout: float = 5.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_requested.set()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("timed out stopping RL WebSocket API")
        self._thread = None
        LOGGER.info("WS_SERVER_STOPPED")

    def publish_status(self, status: RobotStatus) -> None:
        with self._lock:
            self._status = status

    def command_snapshot(self) -> tuple[RobotCommand, int]:
        with self._lock:
            return self._command, self._command_generation

    def pop_dance_request(self) -> str | None:
        with self._lock:
            return self._dance_requests.popleft() if self._dance_requests else None

    def pop_fsm_action(self) -> str | None:
        """Pop one validated, edge-triggered A/X/Y state-machine action."""
        with self._lock:
            return self._fsm_requests.popleft() if self._fsm_requests else None

    def release_control(self) -> None:
        """Return command interpretation to the original gamepad FSM."""
        with self._lock:
            previous_owner = self._owner
            self._owner = None
            self._command = RobotCommand()
            self._command_generation += 1
            self._dance_requests.clear()
            self._fsm_requests.clear()
        LOGGER.info("CONTROL_RELEASE owner=%s", previous_owner)

    def clear_dance_lease(self) -> None:
        """Mark the active dance as explicitly exited by A/gamepad control."""
        with self._lock:
            previous_owner = self._dance_lease_owner
            self._dance_lease_owner = None
        if previous_owner is not None:
            LOGGER.info("DANCE_LEASE_RELEASE client=%d", previous_owner)

    def _accept_command(
        self, command: RobotCommand, client_id: int
    ) -> tuple[bool, str, str, RobotStatus]:
        payload = json.dumps(
            command.to_dict(), separators=(",", ":"), ensure_ascii=False
        )
        with self._lock:
            status = self._status
            previous_dance = self._client_dance_modes.get(client_id, "none")
            if command.fsm_action != "none":
                capability = {
                    "a": status.can_accept_a,
                    "x": status.can_accept_x,
                    "y": status.can_accept_y,
                }[command.fsm_action]
                if not capability:
                    code = "invalid_state"
                    message = (
                        f"FSM action {command.fsm_action.upper()} isn't allowed in "
                        f"the actual state {status.fsm_state}"
                    )
                    LOGGER.warning(
                        "WS_FSM_ACTION client=%d action=%s rejected=%s state=%s",
                        client_id,
                        command.fsm_action,
                        code,
                        status.fsm_state,
                    )
                    return False, code, message, status

                # State actions are edges, not persistent control leases. They
                # also clear any velocity/dance command before the FSM applies
                # the transition on its next 50-Hz cycle.
                self._client_dance_modes[client_id] = "none"
                if command.fsm_action == "a":
                    self._dance_lease_owner = None
                self._owner = None
                self._command = RobotCommand()
                self._command_generation += 1
                self._dance_requests.clear()
                self._fsm_requests.append(command.fsm_action)
                LOGGER.info(
                    "WS_FSM_ACTION client=%d action=%s accepted state=%s",
                    client_id,
                    command.fsm_action,
                    status.fsm_state,
                )
                return (
                    True,
                    "ok",
                    f"FSM action {command.fsm_action.upper()} queued",
                    status,
                )

            if command.ctl_mode == "none":
                self._client_dance_modes[client_id] = "none"
                if self._owner in (None, client_id):
                    self._owner = None
                    self._command = command
                    self._command_generation += 1
                LOGGER.info("WS_COMMAND client=%d accepted payload=%s", client_id, payload)
                return True, "ok", "command accepted", status

            if command.loco_mode != "loco_stop":
                if command.dance_mode != "none":
                    if not status.can_accept_dance:
                        code = "invalid_state"
                        message = (
                            "dance requires the actual FSM state HYBRID with "
                            "a zero velocity command; current state is "
                            f"{status.fsm_state}"
                        )
                        LOGGER.warning(
                            "WS_COMMAND client=%d rejected=%s payload=%s",
                            client_id,
                            code,
                            payload,
                        )
                        return False, code, message, status
                    if command.dance_mode == previous_dance:
                        code = "duplicate_action"
                        message = (
                            f"{command.dance_mode} is already latched; send "
                            "dance_mode=none before requesting it again"
                        )
                        LOGGER.warning(
                            "WS_COMMAND client=%d rejected=%s payload=%s",
                            client_id,
                            code,
                            payload,
                        )
                        return False, code, message, status
                elif not status.can_accept_velocity:
                    code = "invalid_state"
                    message = (
                        "velocity control requires the actual FSM state HYBRID; "
                        f"current state is {status.fsm_state}"
                    )
                    LOGGER.warning(
                        "WS_COMMAND client=%d rejected=%s payload=%s",
                        client_id,
                        code,
                        payload,
                    )
                    return False, code, message, status

            self._client_dance_modes[client_id] = command.dance_mode
            self._owner = client_id
            self._command = command
            self._command_generation += 1
            if command.dance_mode != "none":
                self._dance_lease_owner = client_id
                self._dance_requests.append(command.dance_mode)
                LOGGER.info(
                    "DANCE_QUEUED client=%d dance=%s",
                    client_id,
                    command.dance_mode,
                )
            LOGGER.info("WS_COMMAND client=%d accepted payload=%s", client_id, payload)
            return True, "ok", "command accepted", status

    def _release_client(self, client_id: int) -> None:
        stop_dance = False
        owned_control = False
        with self._lock:
            self._client_dance_modes.pop(client_id, None)
            if self._owner == client_id:
                owned_control = True
                # A disconnected API client cannot leave velocity commands
                # latched. The next FSM cycle returns to neutral gamepad input.
                self._owner = None
                self._command = RobotCommand()
                self._command_generation += 1
                self._dance_requests.clear()
            if self._dance_lease_owner == client_id:
                self._dance_lease_owner = None
                if self._status.fsm_state in ("go_to_mimic_start", "mimic"):
                    # This internal request survives the short Mimic entry
                    # interpolation, then follows the normal A exit path.
                    self._fsm_requests.append("stand")
                    stop_dance = True
        LOGGER.info(
            "WS_DISCONNECT client=%d release_velocity=%s stop_dance=%s",
            client_id,
            owned_control,
            stop_dance,
        )

    def _status_json(self) -> str:
        with self._lock:
            payload = self._status.to_dict()
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def _command_result_json(
        self,
        success: bool,
        code: str,
        message: str,
        status: RobotStatus | None = None,
    ) -> str:
        if status is None:
            with self._lock:
                status = self._status
        payload = {
            "type": "command_result",
            "success": success,
            "code": code,
            "message": message,
            "fsm_state": status.fsm_state,
            "active_policy": status.active_policy,
            "dance_mode": status.dance_mode,
            "can_accept_velocity": status.can_accept_velocity,
            "can_accept_dance": status.can_accept_dance,
            "can_accept_a": status.can_accept_a,
            "can_accept_x": status.can_accept_x,
            "can_accept_y": status.can_accept_y,
            "recovery_active": status.recovery_active,
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    async def _send_statuses(self, websocket: ServerConnection) -> None:
        interval = 1.0 / self.status_hz
        while True:
            await websocket.send(self._status_json())
            await asyncio.sleep(interval)

    async def _handler(self, websocket: ServerConnection) -> None:
        client_id = next(self._client_ids)
        LOGGER.info(
            "WS_CONNECTED client=%d remote=%s", client_id, websocket.remote_address
        )
        authenticated = False
        sender: asyncio.Task[None] | None = None
        watchdog: asyncio.Task[None] | None = None
        last_activity = time.monotonic()

        async def monitor_heartbeat() -> None:
            interval = min(0.25, self.heartbeat_timeout / 4.0)
            while True:
                await asyncio.sleep(interval)
                age = time.monotonic() - last_activity
                if age > self.heartbeat_timeout:
                    LOGGER.warning(
                        "WS_HEARTBEAT_TIMEOUT client=%d age=%.3f timeout=%.3f",
                        client_id,
                        age,
                        self.heartbeat_timeout,
                    )
                    await websocket.close(code=1011, reason="heartbeat timeout")
                    return
        try:
            raw_auth = await asyncio.wait_for(
                websocket.recv(), timeout=self.auth_timeout
            )
            try:
                auth_value = json.loads(raw_auth)
            except (json.JSONDecodeError, TypeError):
                auth_value = None
            if auth_value != self.auth_token:
                LOGGER.warning("WS_AUTH_FAILED client=%d", client_id)
                await websocket.close(code=1008, reason="authentication failed")
                return

            authenticated = True
            LOGGER.info("WS_AUTH_SUCCESS client=%d", client_id)
            await websocket.send(json.dumps(AUTH_RESPONSE))
            sender = asyncio.create_task(self._send_statuses(websocket))
            watchdog = asyncio.create_task(monitor_heartbeat())

            async for message in websocket:
                try:
                    if isinstance(message, bytes):
                        raise ProtocolError("binary messages are not supported")
                    payload = json.loads(message)
                    last_activity = time.monotonic()
                    if (
                        isinstance(payload, dict)
                        and payload.get("type") == "heartbeat"
                        and set(payload) == {"type"}
                    ):
                        continue
                    command = RobotCommand.from_payload(payload)
                    result = self._accept_command(command, client_id)
                    await websocket.send(self._command_result_json(*result))
                except (json.JSONDecodeError, ProtocolError) as exc:
                    LOGGER.warning(
                        "WS_COMMAND_INVALID client=%d error=%s", client_id, exc
                    )
                    await websocket.send(
                        self._command_result_json(
                            False, "invalid_request", str(exc)
                        )
                    )
        except (asyncio.TimeoutError, ConnectionClosed):
            pass
        finally:
            if sender is not None:
                sender.cancel()
                with suppress(asyncio.CancelledError, ConnectionClosed):
                    await sender
            if watchdog is not None:
                watchdog.cancel()
                with suppress(asyncio.CancelledError, ConnectionClosed):
                    await watchdog
            if authenticated:
                self._release_client(client_id)
            LOGGER.info("WS_CLOSED client=%d authenticated=%s", client_id, authenticated)

    async def _serve(self) -> None:
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=64 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ) as server:
            sockets = server.sockets
            if sockets:
                self.bound_port = int(sockets[0].getsockname()[1])
            self._started.set()
            while not self._stop_requested.is_set():
                await asyncio.sleep(0.05)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._startup_error = exc
            LOGGER.exception("RL WebSocket server failed")
            self._started.set()
