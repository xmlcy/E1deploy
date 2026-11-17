"""Synchronous and asynchronous clients for secondary development."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
import inspect
import json
import threading
import time
from types import TracebackType
from typing import Any, Callable, Dict

from websockets.asyncio.client import ClientConnection as AsyncConnection
from websockets.asyncio.client import connect as async_connect
from websockets.sync.client import ClientConnection as SyncConnection
from websockets.sync.client import connect as sync_connect

from .models import CommandResult, RobotCommand, RobotStatus


class E1SdkError(RuntimeError):
    """Connection, authentication, or protocol error."""


def _connect_options(connect_function: Callable[..., Any]) -> Dict[str, Any]:
    """Disable proxies when supported while remaining compatible with v13.

    ``proxy=`` was added to websockets after the Python 3.8-compatible v13
    release. Passing it through v13's ``**kwargs`` reaches the socket layer and
    fails with an unexpected-keyword error.
    """
    options: Dict[str, Any] = {"compression": None}
    if "proxy" in inspect.signature(connect_function).parameters:
        options["proxy"] = None
    return options


def _decode_message(message: str | bytes) -> CommandResult | RobotStatus:
    if isinstance(message, bytes):
        raise E1SdkError("server returned an unexpected binary message")
    try:
        payload = json.loads(message)
        if not isinstance(payload, dict):
            raise ValueError("response isn't a JSON object")
        if payload.get("type") == "command_result" or "error" in payload:
            return CommandResult.from_dict(payload)
        return RobotStatus.from_dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise E1SdkError(f"invalid server response: {exc}") from exc


class E1RobotClient:
    """Blocking client suitable for scripts and non-async applications."""

    def __init__(
        self,
        uri: str = "ws://192.168.254.100:8765",
        *,
        auth_token: str = "droidpad",
        open_timeout: float = 5.0,
        close_timeout: float = 1.0,
        heartbeat_interval: float = 0.5,
    ) -> None:
        if heartbeat_interval <= 0.0:
            raise ValueError("heartbeat_interval must be positive")
        self.uri = uri
        self.auth_token = auth_token
        self.open_timeout = open_timeout
        self.close_timeout = close_timeout
        self.heartbeat_interval = float(heartbeat_interval)
        self._connection: SyncConnection | None = None
        self._send_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._pending_statuses: deque[RobotStatus] = deque(maxlen=100)
        self.last_command_result: CommandResult | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def connect(self) -> E1RobotClient:
        if self._connection is not None:
            return self
        connection = sync_connect(
            self.uri,
            open_timeout=self.open_timeout,
            close_timeout=self.close_timeout,
            **_connect_options(sync_connect),
        )
        try:
            connection.send(json.dumps(self.auth_token))
            response = connection.recv(timeout=self.open_timeout)
            if isinstance(response, bytes) or json.loads(response) != "droidup":
                raise E1SdkError(f"authentication failed: {response!r}")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._pending_statuses.clear()
        self.last_command_result = None
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="e1-sdk-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return self

    def close(self) -> None:
        self._heartbeat_stop.set()
        heartbeat_thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if (
            heartbeat_thread is not None
            and heartbeat_thread is not threading.current_thread()
        ):
            heartbeat_thread.join(timeout=self.heartbeat_interval + 0.5)
        connection = self._connection
        self._connection = None
        if connection is not None:
            with self._send_lock:
                connection.close()
        self._pending_statuses.clear()

    def _heartbeat_loop(self) -> None:
        payload = json.dumps({"type": "heartbeat"}, separators=(",", ":"))
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            connection = self._connection
            if connection is None:
                return
            try:
                with self._send_lock:
                    connection.send(payload)
            except Exception:
                return

    def send_command(
        self, command: RobotCommand, timeout: float = 2.0
    ) -> CommandResult:
        connection = self._require_connection()
        with self._send_lock:
            connection.send(
                json.dumps(
                    command.to_dict(), separators=(",", ":"), ensure_ascii=False
                )
            )

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise E1SdkError("timed out waiting for command acknowledgement")
            try:
                response = connection.recv(timeout=remaining)
            except TimeoutError as exc:
                raise E1SdkError(
                    "timed out waiting for command acknowledgement"
                ) from exc
            decoded = _decode_message(response)
            if isinstance(decoded, CommandResult):
                self.last_command_result = decoded
                return decoded
            self._pending_statuses.append(decoded)

    def recv_status(self, timeout: float | None = None) -> RobotStatus:
        if self._pending_statuses:
            return self._pending_statuses.popleft()

        connection = self._require_connection()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.0:
                raise E1SdkError("timed out waiting for robot status")
            try:
                response = connection.recv(timeout=remaining)
            except TimeoutError as exc:
                raise E1SdkError("timed out waiting for robot status") from exc
            decoded = _decode_message(response)
            if isinstance(decoded, RobotStatus):
                return decoded
            # Command rejections are data, not malformed status frames.  Keep
            # the latest result available and continue to the next status.
            self.last_command_result = decoded

    def stand(
        self, timeout: float = 2.0, *, transition_timeout: float = 5.0
    ) -> CommandResult:
        # Standing is also the SDK-level way to leave a running dance. Mimic
        # must exit through A; never bypass that deterministic FSM transition.
        released = self.release(timeout)
        if not released.success:
            return released
        if released.fsm_state == "mimic":
            switched = self.press_a(timeout)
            if not switched.success:
                return switched
            self._wait_for_hybrid(transition_timeout)
        return self.send_command(RobotCommand.stand(), timeout)

    def walk(
        self,
        vel_x: float = 0.0,
        vel_y: float = 0.0,
        vel_yaw: float = 0.0,
        *,
        timeout: float = 2.0,
    ) -> CommandResult:
        return self.send_command(RobotCommand.walk(vel_x, vel_y, vel_yaw), timeout)

    def damping(self, timeout: float = 2.0) -> CommandResult:
        return self.send_command(RobotCommand.damping(), timeout)

    def release(self, timeout: float = 2.0) -> CommandResult:
        return self.send_command(RobotCommand.release(), timeout)

    def press_a(self, timeout: float = 2.0) -> CommandResult:
        return self.send_command(RobotCommand.press_a(), timeout)

    def press_x(self, timeout: float = 2.0) -> CommandResult:
        return self.send_command(RobotCommand.press_x(), timeout)

    def press_y(self, timeout: float = 2.0) -> CommandResult:
        return self.send_command(RobotCommand.press_y(), timeout)

    def _wait_for_hybrid(
        self, timeout: float, *, dance_ready: bool = False
    ) -> RobotStatus:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise E1SdkError(
                    "timed out waiting for the robot to enter Hybrid"
                )
            status = self.recv_status(timeout=remaining)
            ready = status.can_accept_dance if dance_ready else status.can_accept_velocity
            if status.fsm_state == "hybrid" and ready:
                return status

    def dance(
        self,
        number: int,
        timeout: float = 2.0,
        *,
        force: bool = False,
        transition_timeout: float = 5.0,
    ) -> CommandResult:
        dance_command = RobotCommand.dance(number)
        # Clear the edge latch before requesting a new dance action.
        released = self.release(timeout)
        if not released.success:
            return released
        if force:
            if released.fsm_state != "hybrid":
                switched = self.press_a(timeout)
                if not switched.success:
                    return switched
            if released.fsm_state != "hybrid" or not released.can_accept_dance:
                self._wait_for_hybrid(transition_timeout, dance_ready=True)
        return self.send_command(dance_command, timeout)

    def _require_connection(self) -> SyncConnection:
        if self._connection is None:
            raise E1SdkError("client isn't connected")
        return self._connection

    def __enter__(self) -> E1RobotClient:
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncE1RobotClient:
    """Asyncio client suitable for services and event-driven applications."""

    def __init__(
        self,
        uri: str = "ws://192.168.254.100:8765",
        *,
        auth_token: str = "droidpad",
        open_timeout: float = 5.0,
        close_timeout: float = 1.0,
        heartbeat_interval: float = 0.5,
    ) -> None:
        if heartbeat_interval <= 0.0:
            raise ValueError("heartbeat_interval must be positive")
        self.uri = uri
        self.auth_token = auth_token
        self.open_timeout = open_timeout
        self.close_timeout = close_timeout
        self.heartbeat_interval = float(heartbeat_interval)
        self._connection: AsyncConnection | None = None
        self._send_lock: asyncio.Lock | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._pending_statuses: deque[RobotStatus] = deque(maxlen=100)
        self.last_command_result: CommandResult | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    async def connect(self) -> AsyncE1RobotClient:
        if self._connection is not None:
            return self
        connection = await async_connect(
            self.uri,
            open_timeout=self.open_timeout,
            close_timeout=self.close_timeout,
            **_connect_options(async_connect),
        )
        try:
            await connection.send(json.dumps(self.auth_token))
            response = await connection.recv()
            if isinstance(response, bytes) or json.loads(response) != "droidup":
                raise E1SdkError(f"authentication failed: {response!r}")
        except BaseException:
            await connection.close()
            raise
        self._connection = connection
        self._send_lock = asyncio.Lock()
        self._pending_statuses.clear()
        self.last_command_result = None
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def close(self) -> None:
        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
        connection = self._connection
        self._connection = None
        if connection is not None:
            send_lock = self._send_lock
            if send_lock is None:
                await connection.close()
            else:
                async with send_lock:
                    await connection.close()
        self._send_lock = None
        self._pending_statuses.clear()

    async def _heartbeat_loop(self) -> None:
        payload = json.dumps({"type": "heartbeat"}, separators=(",", ":"))
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            connection = self._connection
            send_lock = self._send_lock
            if connection is None or send_lock is None:
                return
            try:
                async with send_lock:
                    await connection.send(payload)
            except Exception:
                return

    async def _receive_message(
        self, timeout: float | None
    ) -> CommandResult | RobotStatus:
        receive = self._require_connection().recv()
        raw = await receive if timeout is None else await asyncio.wait_for(
            receive, timeout
        )
        return _decode_message(raw)

    async def send_command(
        self, command: RobotCommand, timeout: float = 2.0
    ) -> CommandResult:
        connection = self._require_connection()
        send_lock = self._send_lock
        if send_lock is None:
            raise E1SdkError("client heartbeat isn't initialized")
        async with send_lock:
            await connection.send(
                json.dumps(
                    command.to_dict(), separators=(",", ":"), ensure_ascii=False
                )
            )

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise E1SdkError("timed out waiting for command acknowledgement")
            try:
                decoded = await self._receive_message(remaining)
            except asyncio.TimeoutError as exc:
                raise E1SdkError(
                    "timed out waiting for command acknowledgement"
                ) from exc
            if isinstance(decoded, CommandResult):
                self.last_command_result = decoded
                return decoded
            self._pending_statuses.append(decoded)

    async def recv_status(self, timeout: float | None = None) -> RobotStatus:
        if self._pending_statuses:
            return self._pending_statuses.popleft()

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.0:
                raise E1SdkError("timed out waiting for robot status")
            try:
                decoded = await self._receive_message(remaining)
            except asyncio.TimeoutError as exc:
                raise E1SdkError("timed out waiting for robot status") from exc
            if isinstance(decoded, RobotStatus):
                return decoded
            self.last_command_result = decoded

    async def stand(
        self, timeout: float = 2.0, *, transition_timeout: float = 5.0
    ) -> CommandResult:
        released = await self.release(timeout)
        if not released.success:
            return released
        if released.fsm_state == "mimic":
            switched = await self.press_a(timeout)
            if not switched.success:
                return switched
            await self._wait_for_hybrid(transition_timeout)
        return await self.send_command(RobotCommand.stand(), timeout)

    async def walk(
        self,
        vel_x: float = 0.0,
        vel_y: float = 0.0,
        vel_yaw: float = 0.0,
        *,
        timeout: float = 2.0,
    ) -> CommandResult:
        return await self.send_command(
            RobotCommand.walk(vel_x, vel_y, vel_yaw), timeout
        )

    async def damping(self, timeout: float = 2.0) -> CommandResult:
        return await self.send_command(RobotCommand.damping(), timeout)

    async def release(self, timeout: float = 2.0) -> CommandResult:
        return await self.send_command(RobotCommand.release(), timeout)

    async def press_a(self, timeout: float = 2.0) -> CommandResult:
        return await self.send_command(RobotCommand.press_a(), timeout)

    async def press_x(self, timeout: float = 2.0) -> CommandResult:
        return await self.send_command(RobotCommand.press_x(), timeout)

    async def press_y(self, timeout: float = 2.0) -> CommandResult:
        return await self.send_command(RobotCommand.press_y(), timeout)

    async def _wait_for_hybrid(
        self, timeout: float, *, dance_ready: bool = False
    ) -> RobotStatus:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise E1SdkError(
                    "timed out waiting for the robot to enter Hybrid"
                )
            status = await self.recv_status(timeout=remaining)
            ready = status.can_accept_dance if dance_ready else status.can_accept_velocity
            if status.fsm_state == "hybrid" and ready:
                return status

    async def dance(
        self,
        number: int,
        timeout: float = 2.0,
        *,
        force: bool = False,
        transition_timeout: float = 5.0,
    ) -> CommandResult:
        dance_command = RobotCommand.dance(number)
        released = await self.release(timeout)
        if not released.success:
            return released
        if force:
            if released.fsm_state != "hybrid":
                switched = await self.press_a(timeout)
                if not switched.success:
                    return switched
            if released.fsm_state != "hybrid" or not released.can_accept_dance:
                await self._wait_for_hybrid(
                    transition_timeout, dance_ready=True
                )
        return await self.send_command(dance_command, timeout)

    def _require_connection(self) -> AsyncConnection:
        if self._connection is None:
            raise E1SdkError("client isn't connected")
        return self._connection

    async def __aenter__(self) -> AsyncE1RobotClient:
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
