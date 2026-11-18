from e1_rl_sdk import E1RobotClient

with E1RobotClient("ws://localhost:8765", auth_token="droidpad") as robot:
    status = robot.recv_status(timeout=3.0)
    print("当前状态:", status.fsm_state)

    result = robot.press_x()
    if not result.success:
        raise RuntimeError(f"X 被拒绝: {result.message}")

    # 等 X 的默认姿态转换完成
    while True:
        status = robot.recv_status(timeout=3.0)
        if status.fsm_state == "standby" and not status.transitioning:
            break

    # 如果机器人倒地，先人工扶正；A 只会在倾角合格时被接受
    while True:
        status = robot.recv_status(timeout=3.0)
        if status.fsm_state == "standby" and status.can_accept_a:
            break

    result = robot.press_a()
    if not result.success:
        raise RuntimeError(f"A 被拒绝: {result.message}")

    # 等实际进入可接收速度命令的状态
    while True:
        status = robot.recv_status(timeout=3.0)
        if status.fsm_state == "hybrid" and status.can_accept_velocity:
            break

    result = robot.walk(vel_x=0.2)
    print("行走命令:", result)