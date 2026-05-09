# Multi-Tenant Fault Tolerance Emulation

这个目录包含了多租户调度系统的容错性模拟实验。

## 实验场景

模拟 3 个 job（A、B、C）在共享集群上的调度：
- **Job A**: 长任务（generate 320s, train 130s），在 iter=2 时 generate 阶段会 crash（50%处），然后恢复
- **Job B**: 短任务（generate 330s, train 120s）
- **Job C**: 短任务（generate 330s, train 120s）

时间参数基于 case4 真实workload，使用 **time_scale=200** 加速到实际执行时间：
- A generate: 320s / 200 = 1.6s
- A train: 130s / 200 = 0.65s
- B/C generate: 330s / 200 = 1.65s
- B/C train: 120s / 200 = 0.6s

**预期行为**：
1. 初始阶段：3 个 job 交织执行，Trainer 行呈现 (A → B → C) 的循环 pattern
2. A crash 后（~5s）：B 和 C 继续交替执行，A 的 Rollout-1 通道空闲
3. A 恢复后（~11s后）：3 个 job 重新交织执行

## 快速运行

### 1. 环境准备

```bash
# 进入项目根目录
cd /Users/wuty/Desktop/ROLL

# 激活虚拟环境
source .venv/bin/activate

# 确保依赖已安装（如果之前没装过）
uv pip install redis codetiming psutil transformers matplotlib
```

### 2. 运行实验

```bash
# 设置 PYTHONPATH 并运行
PYTHONPATH=/Users/wuty/Desktop/ROLL python roll/multi_tenant/emu_tenant.py
```

实验会自动：
1. 启动 Redis 和 scheduler
2. 启动 3 个 job（A、B、C）
3. 模拟 A 在 iter=1 时 crash
4. 10 秒后恢复 A
5. 等待所有 job 完成
6. 生成 timeline 可视化图

### 3. 查看结果

结果输出在 `roll/multi_tenant/emu_failure_demo/` 目录：

```bash
ls roll/multi_tenant/emu_failure_demo/

# 输出文件：
# - A.jsonl, B.jsonl, C.jsonl  # 各 job 的事件日志
# - scheduler.log              # 调度器日志
# - timeline.png               # 时间线可视化图
# - manifest.json              # 元信息（crash 和恢复时间等）
```

**查看 timeline**：
```bash
open roll/multi_tenant/emu_failure_demo/timeline.png
```

Timeline 解释：
- **Rollout-1/2/3 行**：各 job 的 generate 阶段（橙色=A，蓝色=B，灰色=C）
- **Trainer 行**：所有 job 共享的 train 阶段
- **红色虚线**：A crash 时刻
- **蓝色虚线**：A 恢复时刻
- **Hatch pattern**：用于区分不同 job（XXXX=A, ...=B, ---=C）
- **颜色深浅**：浅色=rollout，深色=train

图表采用论文发表风格：
- 使用 seaborn 配色方案和 hatch pattern
- 字体大小 14，适合论文插图
- 同时生成 PNG（展示用）和 PDF（发表用）

## 关键文件说明

### `emu_tenant.py`
主入口脚本，负责：
- 启动 scheduler 和 3 个 job
- 注入 failure 和 recovery 事件
- 生成 timeline 可视化

### `rlvr_pipeline.py`
模拟 pipeline，关键修复：
- **修复前问题**：`section_enter` 在 `scheduler_section()` 之前记录，包含了等待调度的时间
- **修复后**：`section_enter` 移到 `with scheduler_section()` 内部，只记录真正执行的时间

### `scheduler.py`
调度器，优化了轮询延迟：
- 将主循环 sleep 从 0.5s 降到 0.02s
- 移除了执行事件后的无条件 sleep

### `base_pipeline.py`
Pipeline 基类，优化了轮询延迟：
- 将 `check_interval` 从 0.2s 降到 0.02s

## 参数调整

如果需要修改实验参数，编辑 `emu_tenant.py` 的 `main()` 函数：

```python
jobs["A"] = launch_job(
    "A",
    generate_real_sec=320,    # generate 阶段真实时长（秒）
    train_real_sec=130,       # train 阶段真实时长（秒）
    failure_iter=2,           # 在哪个 iter crash（从0开始）
    failure_phase="generate", # 在哪个 phase crash
    failure_after_ratio=0.5,  # crash 在 phase 进行到多少比例时发生
    max_iters=8,              # 总共运行几个 iter
)
```

在 `launch_job()` 函数中修改 `time_scale`：
```python
"--time-scale", "200.0",  # 时间缩放倍数
```

`time_scale=200.0` 表示实际执行时间是真实时间的 1/200，例如：
- `generate_real_sec=320` → 实际执行 1.6 秒
- `train_real_sec=130` → 实际执行 0.65 秒

**推荐配置**：
- 如果希望更快的演示：`time_scale=500`（320s → 0.64s）
- 如果希望更细致观察：`time_scale=100`（320s → 3.2s）
- 当前配置（200）在可视化清晰度和运行时间之间取得平衡

## 故障排查

### Redis 端口占用
如果看到 "Address already in use" 错误：
```bash
# 查找占用端口的 redis 进程
ps aux | grep redis-server

# 手动清理
redis-cli -h 127.0.0.1 -p <PORT> shutdown nosave
```

### 依赖缺失
如果看到 `ModuleNotFoundError`：
```bash
# 安装缺失的依赖
uv pip install <module_name>
```

### 查看详细日志
```bash
# 查看调度器日志
cat roll/multi_tenant/emu_failure_demo/scheduler.log

# 查看 job 事件流
cat roll/multi_tenant/emu_failure_demo/A.jsonl | jq .
```

## 技术细节

### 事件流
1. Job 发布 `section_enter` 事件到 Redis pubsub
2. Scheduler 监听事件，根据策略分配资源
3. Scheduler 设置 job status 为 "running"
4. Job 轮询 status，拿到资源后开始执行
5. Job 执行完毕，发布 `section_done` 事件
6. 循环往复

### 调度策略 (FCFS)
- Init 事件优先：需要所有资源空闲
- Generate 事件：需要至少 50% 的 gen 资源（可配置 `RATIO`）
- Train 事件：需要 100% 的 train 资源

### 容错机制
- Job crash 时发布 `crashed` 事件
- Scheduler 清理该 job 的资源分配
- Job 恢复时发布 `recovered` 事件
- Scheduler 将该 job 重新加入调度队列
