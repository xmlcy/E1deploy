# E1 Sim2Real FSM 与 WebSocket API

本项目在原有 50 Hz Hybrid AMP / Mimic 运控循环上实现了接口文档定义的 WebSocket 服务，并提供原始 API 测试脚本与 Python SDK。

运行环境支持 Python 3.8 及以上版本；在 Python 3.8 环境中会使用兼容的 `websockets 13.x`。

## 安全准备

真实机器人执行前应使用吊架并留出安全空间。确认机器人完成上电初始化、手柄连接正常、已扶至站立姿态，并已将外部控制源切换到 WebSocket。`loco_stop` 会进入刚体阻尼模式。

倒地或阻尼后的恢复流程是确定的：按手柄 X 后，机器人从当前位置平滑进入默认关节姿态，并暂时暂停重复倒地判定，方便操作员将机器人拉起。到达默认姿态后必须把机器人扶正再按 A；只有实测机身倾角不超过 30 度且 IMU 有效时才允许进入 Hybrid。角度不正确时 A 会被拒绝，机器人继续保持默认姿态。恢复期间 `fall_latched` 会一直保留，对外状态中的 `recovery_active=true`，直到 A 的角度校验通过。

## 安装与启动

WebSocket 依赖可单独安装：

```bash
python3 -m pip install -r requirements-api.txt
```

如需在其他项目中导入 SDK，可在仓库根目录执行：

```bash
python3 -m pip install -e .
```

启动机器人运控与 API：

```bash
python3 sim2real_e1_21dof_fsm.py \
  --api-host 0.0.0.0 \
  --api-port 8765 \
  --api-auth-token droidpad \
  --api-status-hz 20 \
  --log-file logs/sim2real_fsm.log
```

可用 `--no-api` 禁用服务。`--validate-only` 仍只校验三个 ONNX 模型，不连接硬件或启动 API。

日志默认写入 `logs/sim2real_fsm.log`，单个文件最大 10 MiB，保留 5 个历史文件。日志记录所有收到的 WebSocket 指令、手柄指令变化、控制权变更、FSM 状态切换、安全事件、舞蹈开始/结束/拒绝原因，以及每秒一次的机器人状态快照。可通过 `--log-max-bytes` 和 `--log-backup-count` 调整滚动策略。

```bash
tail -f logs/sim2real_fsm.log
```

## 协议

默认地址是 `ws://192.168.254.100:8765`。连接后首先发送 JSON 字符串 `"droidpad"`，服务端返回 JSON 字符串 `"droidup"`。认证后发送 `robot_cmd`：

```json
{
  "ctl_mode": "loco",
  "vel_x": 0.2,
  "vel_y": 0.0,
  "vel_yaw": 0.0,
  "base_roll": 0.0,
  "base_pitch": 0.0,
  "base_yaw": 0.0,
  "base_z": 0.0,
  "loco_mode": "loco_walk_and_stand",
  "arm_mode": "none"
}
```

服务端以配置频率持续推送文档定义的 `robot_status`。`ctl_mode=none` 只订阅状态；`loco_walk_and_stand` 执行站立/走跑；`loco_stop` 立即进入阻尼。客户端断开后会清空外部命令并返回手柄控制。手柄是原始状态机的最高解释来源：摇杆及 A/X/Y/START/BACK/B 始终优先于当前 WebSocket 命令。

WebSocket 不负责自动推进机器人状态机。速度指令只允许在真实 FSM 状态为 `hybrid` 时发送；舞蹈只允许在真实状态为 `hybrid`，并且策略零速度、实测机身速度/角速度和姿态连续稳定 1 秒后触发。该稳定窗口用于避免从高速跑动直接切舞。每条命令都会收到独立的 `command_result` 确认。状态不满足时，服务端不会锁存或执行命令，而是返回正常的拒绝结果：

```json
{
  "type": "command_result",
  "success": false,
  "code": "invalid_state",
  "message": "dance requires the actual FSM state HYBRID...",
  "fsm_state": "mimic",
  "active_policy": "victory1",
  "dance_mode": "dance1",
  "can_accept_velocity": false,
  "can_accept_dance": false,
  "recovery_active": false
}
```

命令通过校验时同样返回 `type=command_result`、`success=true`、`code=ok`。该确认表示命令已被控制器接受；机器人真实状态仍以持续推送的 `robot_status` 为准。

当前 Hybrid 策略是速度条件策略，因此 `vel_x`、`vel_y`、`vel_yaw` 会真实进入策略，超出模型能力的值会被裁剪到模型训练范围。文档中的 `base_roll`、`base_pitch`、`base_yaw`、`base_z` 会完成类型校验并保持协议兼容，但当前 ONNX 模型没有对应输入，非零值不会改变控制输出。

接口文档说明可用 WebSocket 触发两套舞蹈，但没有定义请求字段。实现增加了可选且向后兼容的字段：

```json
{
  "ctl_mode": "loco",
  "loco_mode": "loco_walk_and_stand",
  "vel_x": 0.0,
  "vel_y": 0.0,
  "vel_yaw": 0.0,
  "dance_mode": "dance1"
}
```

`dance_mode` 可取 `none`、`dance1`、`dance2`。进入舞蹈后 Mimic 策略持续循环运行。退出舞蹈只能先通过手柄或 WebSocket A 返回 Hybrid；Mimic 中的 X、START、BACK 均不会直接改变状态。返回 Hybrid 后才可按 X 或重新选择 Dance 1 / Dance 2。普通 WebSocket 舞蹈请求在 Mimic 状态会收到 `invalid_state`，不会自动切舞；SDK 的 `force=True` 会显式执行并等待 A 流程。

WebSocket 也可发送与手柄语义相同的 A/X/Y 状态动作：

```json
{"ctl_mode":"none","fsm_action":"x"}
```

`fsm_action` 可取 `none`、`a`、`x`、`y`。A 只允许从 Mimic 或已经扶正的 Standby 进入 Hybrid；X 只允许在 IDLE、DAMPING、STANDBY、HYBRID 中执行；Y 是进入阻尼但不退出程序的可恢复急停。B 不对 WebSocket 开放，因为 B 会结束机器人主程序，远程调用后无法再通过 WebSocket 恢复。手柄输入仍拥有最高优先级。

舞蹈请求是边沿触发事件：同一客户端连续周期发送相同舞蹈只接受第一次，SDK 会自动发送复位包。动作成功触发后，WebSocket 的瞬时触发权会释放，但 Mimic 策略会继续运行，直至手柄或 WebSocket A 明确退出。

SDK 每 0.5 秒发送应用层心跳，服务端默认租约超时为 2.5 秒（可通过 `--api-heartbeat-timeout` 配置）。客户端正常退出、崩溃、被强制结束、网络中断或暂停而停止心跳后，服务端会清除外部速度；如果该客户端启动了舞蹈，还会自动排队 A，使机器人退出 Mimic 并回到零速度 Hybrid/Stand。原始 WebSocket 客户端也必须发送 `{"type":"heartbeat"}`。

状态响应除文档字段外还包含真实运行扩展：`fsm_state`、`active_policy`、`dance_mode`、`control_source`、`can_accept_velocity`、`can_accept_dance`、`can_accept_a`、`can_accept_x`、`can_accept_y`、`transitioning` 和 `recovery_active`。这些字段由当前 FSM 和实际活动策略生成，不取自最后一条外部指令。

`joint_q` / `joint_dq` 严格保持文档要求的 10 个元素，顺序为左右腿各自的髋俯仰、髋横滚、髋偏航、膝、踝俯仰；不包含两个踝横滚与腰关节。

## 原始 API 测试

脚本不导入 SDK，直接通过 `websockets` 发送 JSON：

```bash
# 只订阅状态（默认，最安全）
python3 -m examples.raw_api_test --action status

# 原地站立
python3 -m examples.raw_api_test --action stand

# 前进 1 秒，退出前自动发送零速度
python3 -m examples.raw_api_test --action walk --vel-x 0.2 --duration 1

# 阻尼急停
python3 -m examples.raw_api_test --action stop

# 模拟手柄 A / X / Y
python3 -m examples.raw_api_test --action x
python3 -m examples.raw_api_test --action a
python3 -m examples.raw_api_test --action y

# 舞蹈
python3 -m examples.raw_api_test --action dance1
```

开发机测试时可通过 `--uri ws://机器人IP:8765` 指定地址。

## Python SDK

详细的 SDK 类、方法、参数、返回值、安全语义和心跳说明见
[e1_rl_sdk/README.md](e1_rl_sdk/README.md)。

同步接口：

```python
from e1_rl_sdk import E1RobotClient

with E1RobotClient("ws://192.168.254.100:8765") as robot:
    status = robot.recv_status(timeout=2.0)
    print(status.fsm_state, status.bus_voltage)

    result = robot.stand()
    if not result.success:
        print("命令未执行:", result.code, result.message, result.fsm_state)
    else:
        walk_result = robot.walk(vel_x=0.2, vel_y=0.0, vel_yaw=0.0)
        print("行走命令是否接受:", walk_result.success)

    # 与手柄 A/X/Y 相同的状态动作
    print(robot.press_x())
    print(robot.press_a())

    # 若正在跳另一支舞，先 A 回到 Hybrid 并确认真实状态，再跳 dance2
    print(robot.dance(2, force=True))
```

异步接口：

```python
import asyncio
from e1_rl_sdk import AsyncE1RobotClient

async def main():
    async with AsyncE1RobotClient() as robot:
        result = await robot.stand()
        if not result.success:
            print(result.code, result.message, result.fsm_state)
        print(await robot.recv_status())

asyncio.run(main())
```

`stand()`、`walk()`、`dance()`、`damping()`、`release()`、`press_a()`、`press_x()` 和 `press_y()` 都返回 `CommandResult`。Mimic 舞蹈中调用 `stand()` 会先发送 A，等待真实状态回到 Hybrid，再发送零速度站立命令；Damping 等状态不会被自动绕过。`dance(number, force=True)` 会先请求 A，等待真实状态变为可跳舞的 Hybrid，然后再触发目标舞蹈；它不会绕过 A 的状态与倾角检查。机器人处于不可控状态时返回 `success=False`，不会抛出协议异常，也不会中断后续状态接收；连接超时、断线或畸形协议帧仍会抛出 `E1SdkError`。SDK 还提供 `RobotCommand`、`RobotStatus` 和 `CommandResult` 数据类，便于上层程序做类型化封装。

Linux 终端中的 `Ctrl+C` 是中断，`Ctrl+Z` 默认是暂停进程而不是退出。暂停中的客户端不会执行 `with E1RobotClient(...)` 的连接清理，WebSocket 及最后一条控制命令可能暂时保持。示例脚本会将 `Ctrl+Z` 转换为安全退出；二次开发程序也应自行处理 `SIGTSTP`，或只使用 `Ctrl+C`/正常退出。机器人主控制程序同样会拦截 `SIGTSTP`，执行最终阻尼后退出。

## 本地验证

无需机器人硬件即可运行协议、服务端和 SDK 回环测试：

```bash
python3 -m unittest discover -v
python3 sim2real_e1_21dof_fsm.py --validate-only
```
