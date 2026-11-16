"""Synchronous and asynchronous clients for secondary development."""

from __future__ import annotations

import json
import inspect
from types import TracebackType
from typing import Any, Callable, Dict

from websockets.asyncio.client import ClientConnection as AsyncConnection
from websockets.asyncio.client import connect as async_connect
from websockets.sync.client import ClientConnection as SyncConnection
from websockets.sync.client import connect as sync_connect

from .models import RobotCommand, RobotStatus


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


def _decode_status(message: str | bytes) -> RobotStatus:
    if isinstance(message, bytes):
        raise E1SdkError("server returned an unexpected binary message")
    try:
        payload = json.loads(message)
        if not isinstance(payload, dict):
            raise ValueError("status isn't a JSON object")
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
    ) -> None:
        self.uri = uri
        self.auth_token = auth_token
        self.open_timeout = open_timeout
        self.close_timeout = close_timeout
        self._connection: SyncConnection | None = None

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
        return self

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def send_command(self, command: RobotCommand) -> None:
        connection = self._require_connection()
        connection.send(
            json.dumps(command.to_dict(), separators=(",", ":"), ensure_ascii=False)
        )

    def recv_status(self, timeout: float | None = None) -> RobotStatus:
        return _decode_status(self._require_connection().recv(timeout=timeout))

    def stand(self) -> None:
        self.send_command(RobotCommand.stand())

    def walk(self, vel_x: float = 0.0, vel_y: float = 0.0, vel_yaw: float = 0.0) -> None:
        self.send_command(RobotCommand.walk(vel_x, vel_y, vel_yaw))

    def damping(self) -> None:
        self.send_command(RobotCommand.damping())

    def release(self) -> None:
        self.send_command(RobotCommand.release())

    def dance(self, number: int) -> None:
        # Clear the edge latch before requesting a new dance action.
        self.release()
        self.send_command(RobotCommand.dance(number))

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
    ) -> None:
        self.uri = uri
        self.auth_token = auth_token
        self.open_timeout = open_timeout
        self.close_timeout = close_timeout
        self._connection: AsyncConnection | None = None

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
        return self

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def send_command(self, command: RobotCommand) -> None:
        await self._require_connection().send(
            json.dumps(command.to_dict(), separators=(",", ":"), ensure_ascii=False)
        )

    async def recv_status(self) -> RobotStatus:
        return _decode_status(await self._require_connection().recv())

    async def stand(self) -> None:
        await self.send_command(RobotCommand.stand())

    async def walk(
        self, vel_x: float = 0.0, vel_y: float = 0.0, vel_yaw: float = 0.0
    ) -> None:
        await self.send_command(RobotCommand.walk(vel_x, vel_y, vel_yaw))

    async def damping(self) -> None:
        await self.send_command(RobotCommand.damping())

    async def release(self) -> None:
        await self.send_command(RobotCommand.release())

    async def dance(self, number: int) -> None:
        await self.release()
        await self.send_command(RobotCommand.dance(number))

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
