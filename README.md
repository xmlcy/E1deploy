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

WebSocket 不负责自动推进机器人状态机。速度指令只允许在真实 FSM 状态为 `hybrid` 时发送；舞蹈只允许在真实状态为 `hybrid` 且当前策略速度指令为零时触发。状态不满足时，服务端不会锁存或执行命令，而是返回：

```json
{
  "error": {
    "code": "invalid_state",
    "message": "dance requires the actual FSM state HYBRID...",
    "fsm_state": "mimic",
    "active_policy": "victory1",
    "dance_mode": "dance1",
    "can_accept_velocity": false,
    "can_accept_dance": false
  }
}
```

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

`dance_mode` 可取 `none`、`dance1`、`dance2`。进入舞蹈后 Mimic 策略持续循环运行。退出舞蹈只能先按手柄 A 返回 Hybrid；Mimic 中的 X、START、BACK 均不会直接改变状态。返回 Hybrid 后才可按 X 或重新选择 Dance 1 / Dance 2。WebSocket 在 Mimic 状态请求另一舞蹈会收到 `invalid_state`，不会自动切舞。

舞蹈请求是边沿触发事件：同一客户端连续周期发送相同舞蹈只接受第一次，SDK 会自动发送复位包。动作成功触发后，WebSocket 的瞬时触发权会释放，但 Mimic 策略会继续运行，直至手柄 A 明确退出。

状态响应除文档字段外还包含真实运行扩展：`fsm_state`、`active_policy`、`dance_mode`、`control_source`、`can_accept_velocity`、`can_accept_dance`、`transitioning` 和 `recovery_active`。这些字段由当前 FSM 和实际活动策略生成，不取自最后一条外部指令。

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

# 舞蹈
python3 -m examples.raw_api_test --action dance1
```

开发机测试时可通过 `--uri ws://机器人IP:8765` 指定地址。

## Python SDK

同步接口：

```python
from e1_rl_sdk import E1RobotClient

with E1RobotClient("ws://192.168.254.100:8765") as robot:
    robot.stand()
    status = robot.recv_status(timeout=2.0)
    print(status.loco_mode, status.bus_voltage)

    robot.walk(vel_x=0.2, vel_y=0.0, vel_yaw=0.0)
    robot.stand()
    robot.dance(1)
    # robot.damping()  # 刚体阻尼急停
```

异步接口：

```python
import asyncio
from e1_rl_sdk import AsyncE1RobotClient

async def main():
    async with AsyncE1RobotClient() as robot:
        await robot.stand()
        print(await robot.recv_status())

asyncio.run(main())
```

SDK 还提供 `RobotCommand` 和 `RobotStatus` 数据类，便于上层程序做类型化封装。

## 本地验证

无需机器人硬件即可运行协议、服务端和 SDK 回环测试：

```bash
python3 -m unittest discover -v
python3 sim2real_e1_21dof_fsm.py --validate-only
```
