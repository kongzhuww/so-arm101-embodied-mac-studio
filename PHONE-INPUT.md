# 手机输入验证与6号单步接口

## 已验证

- realme RMX8899 / Android16，Termux可连接，热点手机10.19.220.45，Mac en1为10.19.220.84。
- Edge155虽报告支持AR，但实际requestSession反复NotSupportedError；改local参考空间后仍失败。
- 用户安装的Chrome156成功创建immersive-ar/local会话，Mac收到707条位姿消息。
- 验证快照：reports/phone-input-verified-before-motor6.json。
- 手机输入服务：scripts/phone_input_probe.py；https://10.19.220.84:4443/；20分钟自动退出。无机械臂连接。
- 原厂teleop前端在本地响应内替换local-floor为local，并精简可选特性；未修改已安装包。
- 最新只读机械臂检查：六轴位置2028,1226,2528,2846,1962,2415，均torque0。6号speed50/accel1/torque_limit50，temperature38，status0。

## 首次手机触发6号：已实机通过

用户本次明确同意后，手机新按钮触发了一次6号运动：起点2415，目标2447，Reached=True；日志motion-tests/phone6-20261031T170217-501460Z.jsonl。退出关闭6号扭矩。随后独立read_only_probe确认六轴torque0，1..5位置未变，6号最终2445（相对起点+30格，约2.6度）。串口已关闭；不会继续响应按钮。本次只验证单次按钮到6号，非持续夹爪或整臂遥操作。

## 单次按钮接口

scripts/phone_gripper_once.py默认计划模式，无串口访问。

经当前用户明确同意后，执行入口为：

```
bash run python -B -u scripts/phone_gripper_once.py --execute --ack-motor6-motion
```

- 等待新鲜手机数据；要求先观察到A/B都松开，再收到一个新按钮按下。
- A是+32编码格，B是-32编码格；机械开合方向需观察确认，不能据按钮名字推断。
- 仅执行一个小步（舵机编码角约2.8度），然后退出；长按不重复，无自动回位。
- 最多等待60秒，不连接串口直到收到按钮触发。数据超过1秒无更新时拒绝触发。
- 复用motor6_single_step.perform，支持有符号不超过32格的一步；其余五轴只读。
- 保留固件目标写入会隐式启用扭矩的实测假设；不做所谓被动预装目标。
- 固定低RAM设置须已匹配，否则停止，不自动重配或加大出力；运动最多2秒，finally关闭6号扭矩。
- 此单步不是持续按住才运动的死手开关。释放按钮不能撤销已下发的一步目标。
- 手机页面/服务本身仍仅采集输入；只有单独运行上述授权入口才可能运动。
- 115项离线测试通过，首次手机触发6号小步实机通过；不代表连续夹爪、整臂遥操作或ACT采集已经验证。

## 后续边界

再次手机驱动6号前须明确预告并取得该次运动同意；夹爪空载、手离开夹口、臂段稳定支撑、可立即断电。
整臂运动、IK实物坐标/关节零位、全程速度与力、录制、ACT训练和推理均未通过验证，不能直接运行官方自动回零teleoperate.py。

## 位置优先演算与1号测试准备

- SO101官方模型固定上游a758567c3978dfeefe282ede0500085a48fe8f78，models/SO101/source.json记录文件SHA256。
- scripts/phone_arm_dryrun.py已改为位置优先、姿态不约束（orientation_weight=0），并非完整6自由度跟随。保留原0.1mm浮点IK残差、2mm请求位移、1度关节变化检查；未放宽门槛。
- 真实3条手机输入重演全部通过，报告reports/phone-arm-position-only-regression.json。这只证明连续角度的数学目标，不证明编码量化后的实际末端精度或碰撞安全。
- 官方模型中性位报告若干相邻链节碰撞，碰撞过滤/实物碰撞检查未验证。
- 1号只读快照backups/base-settings-20261031T171713-160935Z.json：当前位置2028，旧目标0，torque0，旧speed0/acceleration0/torque_limit1000。禁止直接使能后追旧目标。
- 新增scripts/base_single_step.py，默认计划模式。经此次明确授权后才可用--execute --ack-base-motion：仅1号RAM设置torque_limit50/acceleration1/speed50/time0，第一条目标就是当前位置+16格（约1.4度），可能隐式使能。最多2秒，随后关闭1号扭矩，无重试运动、回位或加大出力，其他轴只读。
- 130项离线测试通过。用户明确选择本次运动后，1号实机测试成功：2028→目标2044，Reached=True；日志motion-tests/base1-20261031T171943-200399Z.jsonl。随后独立读取1号2042（相对起点+14格，约1.23度），六轴torque0，其他轴位置未变，串口关闭。仅1号/6号小步已通过，其他机身轴与多轴跟随仍未验证，未开放整臂控制。
