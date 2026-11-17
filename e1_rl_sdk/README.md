# E1 Robot Python SDK

The SDK wraps the project's authenticated WebSocket API. It provides a blocking
client for scripts and an asyncio client for event-driven applications. Both
clients send an application heartbeat automatically while connected.

## Install and connect

Install the API dependencies in the Python environment used by the application:

```bash
python -m pip install -r requirements-api.txt
```

The SDK expects a running `sim2real_e1_21dof_fsm.py` process with its WebSocket
API enabled. The default endpoint is `ws://192.168.254.100:8765` and the default
token is `droidpad`; use the configured host, port, and token for your robot.
For example:

```python
from e1_rl_sdk import E1RobotClient

with E1RobotClient(
    "ws://192.168.254.100:8765",
    auth_token="droidpad",
) as robot:
    status = robot.recv_status(timeout=2.0)
    print(status.fsm_state, status.active_policy)
```

The context manager closes the WebSocket on normal exit, exceptions, or
`Ctrl+C`. The synchronous client starts a daemon heartbeat thread after
authentication and stops it during `close()`.

## Read robot status

The server pushes status frames at the configured rate (20 Hz by default).
`recv_status(timeout=None)` returns the next status frame. A finite timeout
raises `E1SdkError` if no status arrives in time. Status frames received while a
command acknowledgement is pending are buffered for later `recv_status()`
calls.

`RobotStatus` contains the documented electrical, velocity, and joint fields,
plus actual FSM information:

| Field | Meaning |
| --- | --- |
| `fsm_state` | Current FSM state, such as `hybrid`, `mimic`, `damping`, or `standby` |
| `active_policy` | Policy currently controlling the robot, for example `hybrid` or `victory1` |
| `dance_mode` | Actual active dance (`dance1`, `dance2`, or `none`) |
| `control_source` | Current command source (`gamepad`, `websocket`, `policy`, or `fsm`) |
| `can_accept_velocity` | Whether the current FSM state accepts velocity commands |
| `can_accept_dance` | Whether dance safety checks currently pass |
| `can_accept_a`, `can_accept_x`, `can_accept_y` | Whether each state action is allowed in the current state |
| `transitioning` | Whether a pose transition is in progress |
| `recovery_active` | Whether the assisted fall recovery pose is active |

These values describe the robot, not merely the last requested command. Joint
position and velocity vectors contain 10 values in the documented API order.

## Command acknowledgements and errors

Every SDK command returns `CommandResult`:

```python
result = robot.walk(vel_x=0.2)
if not result.success:
    print(result.code, result.message, result.fsm_state)
```

`success=True` means the server accepted the command against the actual status
it had when validating the request. It does not mean a transition has completed;
read `RobotStatus` to observe the resulting state. A gamepad action arriving
after acknowledgement can still take priority and replace or preempt an API
request. Rejections such as
`invalid_state` and `duplicate_action` are normal results and don't raise.

Connection, authentication, malformed response, and receive timeout errors
raise `E1SdkError`. Local argument validation (for example an unsupported dance
number or a non-finite velocity) raises `ValueError` before anything is sent.

## Synchronous methods

All methods below are available on `E1RobotClient` and return `CommandResult`,
unless described as a status or lifecycle method.

| Method | Behavior |
| --- | --- |
| `stand(timeout=2.0, transition_timeout=5.0)` | Sends zero velocity. If actual state is `mimic`, sends A, waits for actual Hybrid, then sends zero velocity. It doesn't bypass Damping or the upright check. |
| `walk(vel_x=0, vel_y=0, vel_yaw=0, timeout=2.0)` | Sends physical velocity in m/s and rad/s. Values are validated as finite numbers, then clamped to policy limits by the controller. Current forward limit is 1.5 m/s. |
| `dance(number, timeout=2.0, force=False, transition_timeout=5.0)` | Requests `dance1` or `dance2`. With `force=True`, sends A, waits for actual Hybrid and dance readiness, then requests the selected dance. A still must pass the FSM and upright checks. |
| `press_a(timeout=2.0)` | Requests A. In Mimic, exits to Hybrid; in Standby, enters Hybrid only when upright. |
| `press_x(timeout=2.0)` | Requests X using the current FSM's state-specific behavior. In Damping, moves to the default pose for assisted recovery. |
| `press_y(timeout=2.0)` | Requests recoverable damping. It doesn't terminate the robot process. |
| `damping(timeout=2.0)` | Sends `loco_stop`, which enters Damping through the WebSocket command path. |
| `release(timeout=2.0)` | Releases this client's velocity/control command. On its own, it doesn't exit an already-running dance. |
| `send_command(command, timeout=2.0)` | Sends a `RobotCommand` and waits for its acknowledgement. |
| `recv_status(timeout=None)` | Returns the next buffered or newly received `RobotStatus`. |
| `connect()` / `close()` | Opens or closes the authenticated connection and heartbeat. Both are safe to call repeatedly. |

Example with explicit result handling:

```python
from e1_rl_sdk import E1RobotClient

with E1RobotClient("ws://192.168.254.100:8765") as robot:
    status = robot.recv_status(timeout=2.0)
    print("actual:", status.fsm_state, status.active_policy)

    result = robot.walk(vel_x=0.3)
    if result.success:
        print("walk accepted")
    else:
        print("walk rejected:", result.code, result.message)

    # A true stand request also exits Mimic through A when needed.
    result = robot.stand()
```

### Safe dance switching

Normal `dance()` only works when the server reports dance readiness. The FSM
requires actual Hybrid state, a zero policy velocity command, measured low body
velocity and angular velocity, upright posture, and a continuous one-second
settling window. This protects a transition from a high-speed run.

`force=True` means “request the permitted A transition first, then wait and
retry the requested dance after the actual status is dance-ready.” It does not
mean bypass safety checks. For example:

```python
result = robot.dance(2, force=True, transition_timeout=8.0)
if not result.success:
    print("dance not started:", result.code, result.message)
```

If A is disallowed (for example in Damping), the returned result is a rejection
and the dance is not sent. If A is accepted but Hybrid doesn't become ready
within `transition_timeout`, `E1SdkError` is raised and the dance is not sent.

## Async client

`AsyncE1RobotClient` has the same commands and safety behavior. Use it within one
asyncio event loop and don't call synchronous `recv()` methods from that loop.

```python
import asyncio
from e1_rl_sdk import AsyncE1RobotClient


async def main():
    async with AsyncE1RobotClient("ws://192.168.254.100:8765") as robot:
        status = await robot.recv_status(timeout=2.0)
        print(status.fsm_state)

        result = await robot.dance(1, force=True)
        if not result.success:
            print(result.code, result.message)


asyncio.run(main())
```

The async client runs a background asyncio heartbeat task. `close()` cancels the
task and closes the WebSocket.

## Heartbeat and disconnect safety

By default, both SDK clients send `{"type":"heartbeat"}` every 0.5 seconds.
The server's default heartbeat lease is 2.5 seconds, configured on the robot
process with `--api-heartbeat-timeout`. If commands and heartbeats stop, the
server closes that session and clears its velocity command. If that session
started a dance, the server requests the standard A exit path. The FSM completes
its short dance-entry interpolation if necessary, exits Mimic to zero-velocity
Hybrid/Stand, and leaves the policy lease cleared.

The lease is a connection-liveness safeguard, not a replacement for the robot's
fall protection or physical emergency stop. The hand controller keeps priority.
When the entire controller process or robot host stops, this in-process
WebSocket safeguard cannot run. Automatic dance exit returns the FSM to the
zero-velocity Hybrid policy; it doesn't turn the motors off or enter Damping.

For a custom WebSocket client, send authentication token `"droidpad"` after
connecting, wait for `"droidup"`, then send a heartbeat object at least once
within each lease interval:

```json
{"type":"heartbeat"}
```

Any valid command also refreshes the lease. Do not rely on WebSocket protocol
ping frames alone; the application heartbeat is the lease signal.

## Direct RobotCommand use

`RobotCommand` is exported for callers that need full control over the wire
payload. Its helpers include `stand()`, `walk(...)`, `dance(number)`,
`damping()`, `release()`, `press_a()`, `press_x()`, and `press_y()`.
`send_command()` returns the same `CommandResult` as the convenience methods.

Avoid sending state actions together with a velocity command. The protocol
requires `ctl_mode="none"`, zero velocity, and one `fsm_action` among `a`, `x`,
or `y`. B is intentionally not exposed by WebSocket because it exits the
controller process.

## Threading and usage limits

Use one client connection per controlling application and serialize command
sequences. The synchronous heartbeat thread only sends heartbeat frames; SDK
receive methods should have a single reader at a time. The async client likewise
expects one coroutine to consume incoming frames at a time. Multiple concurrent
readers can consume another call's acknowledgement or status.

The server still checks real FSM state on every command. A client timeout or
failed command doesn't imply that the robot changed state; inspect the next
`RobotStatus` and handle the `CommandResult` explicitly.
