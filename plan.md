# PKU Infra Course C1 FlashKDA KDA Team题：实验进展与后续计划（独立文档版）

> 本文档为完整、独立版本：背景、题面摘要、当前数据、结论、剩余工作、两人分工
> 都内联在本文件中，读者无需打开其他 `TASK.md` / `PLAN.md` / `SUMMARY.md` 即可
> 完整理解 C1 FlashKDA team 题的当前状态与下一步。

## 0. 背景与题面摘要

**作业目标（C1 FlashKDA KDA team 题）**：

- 复现：在 GPU 上跑通 FlashKDA 官方 kernel；用 `ncu` / SASS 确认计算主路径为
  SM80 `mma.sync`（即 `HMMA` 指令），而不是 `wgmma` (SM90) 或 `tcgen05` (SM100)。
- 分析：把官方“CHUNK=16、bf16 状态、SM80 MMA”的三个理由各自量化，说明 CHUNK
  改为 32/64 时哪个先破、代价多大；讨论点 1–6 逐条给出“结论+证据”。
- 挑战：从下面三条 SM100 路线中任选一条动手，并对 `fla_kda_ref/` 做正确性对拍，
  与 FlashKDA 本体做性能比较。做不出正收益也算完成——只要论证扎实。

三条可选路线（来自题面原文）：

1. 只换指令不动算法（例如把 `HMMA` 换成 `tcgen05`）；
2. 大 CHUNK + rescale（CHUNK=32 或 64，配合新的 rescale 数值表示）；
3. 并行度重构：多 head 进一个 CTA、persistent kernel、或 2-CTA/cluster。

**题面硬性要求与可选要求的边界**：

- 硬性：必须从三条中任选一条动手；必须对 `fla_kda_ref/` 做正确性对拍；必须
  与 FlashKDA 本体做性能比较；负 speedup 也可接受。
- 可选 / 非强制：实现 persistent kernel、2-CTA/cluster、tcgen05 或其他
  FlashKDA 真实 kernel variant。
- 因此，只要完成上述两项对比（A 路线只需 ISA probe 即可作为“动手”，但
  要替代为“挑战闭环”仍需引入真实对拍与性能表）。

**GPU 环境**：

- 硬件：NVIDIA B300 SXM6 AC，`sm_103`，Blackwell Ultra；
- 工具链：CUDA 13.0（`/usr/local/cuda-13.0/bin/nvcc`），登录节点与计算节点
  各有一份 `venv`（`/tmp/<jobid>/venv`），运行真实 FlashKDA 必须用 `srun` 进入
  计算节点；
- `FlashKDA/` 子模块 pin 在 commit `1ce47ea`，cutlass pin `5c149f5`；
- FLA 仓库 `fla_kda_ref/`（`naive.py` 朴素 PyTorch、`chunk.py` Triton 参照、
  `backends/flash_kda.py` 适配层）pin 在 `a3edffc`；
- 形状：K3 配置是 96 头 × head_dim 128（93 层中 69 层 KDA、24 层全注意力）；
  TP8 部署下每卡头数 = 96/8 = 12；GEMM 侧形状见 assignment 4.5。

## 1. 现在做了什么、得到什么数据、结论

### 1.1 官方 FlashKDA 在 B300 上的复现

已在 NVIDIA B300 SXM6 AC（`sm_103`）上跑通官方 FlashKDA forward，固定测试形状为 `T=8192, H=96, D=128`。

已有证据：

- `flash_kda` 固定长度：约 `0.9991 ms`；
- K1/K2 分解：K1 约 `273 us`，K2 约 `746 us`；
- K1/K2 tensor-pipe elapsed-cycle 占比分别约 `5.63%`、`19.43%`；
- K2 achieved occupancy 约 `9.37%`，grid 约为 `96 CTA`；
- SASS 中有 `HMMA.16816.F32.BF16` 和 `HMMA.16816.F16`，没有 `TCGEN/QMMA`，同时存在 TMA 指令。

相关文件：

- `HWs/assignments/assignment02/team/c1_flashkda/repro_B300/REPRO.md`
- `HWs/assignments/assignment02/team/c1_flashkda/repro_B300/raw/sass_dump.txt`
- `HWs/assignments/assignment02/team/c1_flashkda/repro_B300/raw/b2_tensor_metrics.csv`

结论：官方实现确实在 B300 上运行，但其矩阵乘主路径仍是 SM80 世代 `mma.sync/HMMA`；不能把它说成完整的 SM100 原生 kernel。

### 1.2 路线 A：SM100 `tcgen05` 指令可用性

已编译最小 `tcgen05.mma` probe：

- `sm_100a` 编译成功；
- `sm_103a` 编译成功；
- SASS 中出现 `UTCHMMA`（即 `tcgen05.mma.cta_group::1.kind::f16`）。

相关文件：

- `challenge_a_sm100/tcgen05_mma_probe.cu`
- `challenge_a_sm100/tcgen05_sass_sm100a.txt`
- `challenge_a_sm100/tcgen05_sass_sm103a.txt`

结论：CUDA 13.0 和 B300 工具链能够编码 `tcgen05`。但这只证明 ISA 和编译器可用，不证明 FlashKDA 的 K1/K2 layout 可以直接替换。现有 FlashKDA 核心 tile 为 `CHUNK=16`，而 tcgen05 的 CTA-group-1 BF16/F16 M 维要求为 64 或 128，因此需要重分块、TMEM、descriptor 和 epilogue 设计，不能只替换一个 intrinsic。

### 1.3 路线 B：大 CHUNK 与 rescale 数值探针

已测试 `CHUNK=16/32/64` 的 rescale 表示：

```text
chunk     max|cum|       bf16       fp64        log
16         57.018        True       True        True
32        101.714        False      True        True
64        190.581        False      True        True
```

最坏路径测试中，`CHUNK=32` 需要的 reciprocal 约为 `1.49e44`，已经超过 BF16 可表示范围；`CHUNK=64` 更严重。log-domain 表示在三种 CHUNK 下保持有限。

相关文件：

- `challenge_b1/chunk32_rescale.py`
- `challenge_b1/chunk32_rescale_output.txt`
- `challenge_b1.md`

结论：可以否定“保留现有 BF16 exp 表示、只把 CHUNK 改成 32”的方案。log-domain 原型只证明数值上有可能，尚未构成完整 FlashKDA K1/K2 kernel，也没有端到端性能数据。

### 1.4 路线 C：真实 FlashKDA 的 head-split dispatch

已在真实 `flash_kda.fwd` 路径上把 `H=96` 切成多个 head 分片，分别放入 CUDA streams，再沿 head 维拼接：

```text
parts     ms       speedup    cold_diff    warm_diff
1         1.0824   1.000      0            0
2         1.1436   0.946      4.359e-02   0
4         1.2034   0.899      0            0
```

相关文件：

- `challenge_k2/bench_flashkda_parallel.py`
- `challenge_k2/bench_flashkda_parallel_output.txt`
- `challenge_k2.md`

结论：在当前实现中，把 head 拆成多个独立 FlashKDA dispatch 没有收益，反而慢约 5% 到 10%。`warm_diff=0`，说明热身后的拼接结果与官方单次 FlashKDA 结果一致；`parts=2` 首次调用的 `cold_diff` 仍需单独解释，不能直接当作正确性通过。

重要边界：这不是 persistent kernel，也不是 2-CTA/cluster kernel。它只否定“host 侧拆成多个独立调用”的收益，不能否定真正的 K2 kernel 重构。

### 1.5 当前总判断

根据题面原文（C1 FlashKDA 第三节“挑战”），题面只要求从三条 SM100 路线中任选一条，完成 `fla_kda_ref` 正确性对拍和 FlashKDA 性能比较；**没有强制实现 persistent kernel、2-CTA/cluster、tcgen05 或其他真实 kernel variant**。

现有证据支持以下结论：

1. 官方实现的 MMA 主路径确实是 SM80 世代，但这不等于官方没有使用较新的 TMA 能力。
2. 单纯把 HMMA 换成 tcgen05 的收益缺乏依据，且现有 `CHUNK=16` layout 与 tcgen05 形状不直接匹配。
3. 保持 BF16 exp 表示时，`CHUNK=32/64` 会发生数值溢出。
4. 在现有 FlashKDA 实现外面增加 head-split dispatch 没有性能收益。
5. 当前 A/B/C 仍然是“指令探针、数值探针、dispatch 实验”，还不能宣称已经完成完整的 SM100 FlashKDA kernel 移植。
6. 已有 Amdahl 估算给出只消除可见 tensor-pipe 部分时约 `1.31x` 的乐观端到端上界；这足以支持“只换 tensor instruction 的收益有限”，但不足以否定包含 TMA、warp specialization 和 layout 重构的完整后端。

## 2. 接下来要做什么、为什么要做

### 2.1 必做一：补独立的 kernel 正确性对拍

当前 C 路线只和同一个 FlashKDA 实现的官方调用比较，缺少独立语义 reference。应复用现有 `tests/test_fwd.py` 的 FLA FP64 `fused_recurrent_kda`，同时比较 `out` 和 `final_state`。

最小测试集：

| 类型 | 配置 | 目的 |
|---|---|---|
| fixed | `B=2,T=15,H=1` | 小于一个 16-token chunk 的尾部 |
| fixed | `B=2,T=17,H=12`，BF16 state | 跨 chunk；覆盖 TP8 每卡 12 头 |
| fixed | `B=1,T=33,H=96`，FP32 state | 多 chunk；覆盖原始 K3 头数 |
| varlen | `seq_lens=[15,16,17,33], H=96` | 检查 tail mask、prefix sum 和序列隔离 |
| continuation | 在 token 16/17 切分，传递 `final_state` | 检查 K2 recurrence 和 state layout |

每个 case 都要：

- 检查 `out` 和 `final_state` 是否 finite；
- 与 FLA FP64 reference 比较；
- `out` 误差阈值使用 `0.005`；
- `final_state` 误差阈值使用 `0.006`；
- 多个 seed 重复，保存原始输出。

为什么做：这是目前最容易被追问的漏洞。特别是 `parts=2` 的首次调用有 `4.359e-2` 差异，不能只用热身后的 `warm_diff=0` 掩盖。

### 2.2 必做二：把性能实验变成可审计的统计

在 B300 上对官方 baseline 和 C 路线各跑多个独立进程，报告 median、p5/p95 或标准差，而不是只保留一次均值。至少补充：

- `H=12/24/48/96`；
- `T=15/16/17/33/8192`；
- cold-start 与 warm-start 分开；
- alloc-inclusive 时间与预分配 workspace 时间分开；
- 使用 Nsight Systems 检查多个 streams 是否真的 overlap。

为什么做：当前 `flash_kda.fwd` 每次调用都会申请 workspace，并且 C++ 路径会构造转置后的 beta buffer。head split 会重复这些开销，因此必须确认负收益来自 kernel、分配、同步还是 `torch.cat`。

注意：不需要为了本题额外寻找真实 SM80 GPU，除非要声称跨硬件代际性能比较。B300 上的 SASS 已足够证明当前 binary 使用 SM80 世代 HMMA 路径。

### 2.3 可选增强：真实 FlashKDA kernel variant

题面没有强制实现 persistent kernel、单 CTA 多 head、2-CTA/cluster 或真实 tcgen05 FlashKDA kernel。因此它不是当前挑战闭环的必做项。

如果时间充足，或老师明确要求“动手”必须落到 CUDA kernel，可以再选择一个最小 K2 variant，例如多 head/warp group 的 CTA 组织或 persistent work loop。该增强必须交付新的 CUDA/CUTLASS kernel 或 launch geometry，以及 correctness、SASS、NCU resource/occupancy 和端到端性能。

如果被 tcgen05 layout、TMEM、CUTLASS API 或 B300 架构限制阻塞，保存完整编译错误，并把结论限定为“该移植路径在当前工具链/布局下无法完成”。在没有这个增强时，答辩中必须明确：C 路线只证明 dispatch-level head split 无收益，不能声称 persistent/2-CTA 已被验证或否定。

### 2.4 可选实验

如果能取得真实 K3 checkpoint 或推理 activation，再补：

- 各层/各头 `A_log`、`dt_bias` 和 gate activation 分布；
- `CHUNK=16/32` 下 `|cumsum|` 的分位数；
- BF16 exp 溢出比例。

拿不到真实 checkpoint 时，不要把当前随机输入实验称为“真实 K3 分布”，统一称为“K3-style 合成最坏界探针”。

### 2.5 不建议继续做的实验

- 不建议只做一个孤立的 tcgen05 指令吞吐 microbenchmark；它不能回答 FlashKDA layout 和端到端收益问题。
- 不建议直接把 `CHUNK` 常量改成 32；这会同时牵涉 K1/K2 layout、workspace、TMA descriptor、Neumann inverse 和数值表示。
- 不建议再扩展更多 Python CUDA streams；当前数据已经显示 dispatch 拆分没有收益。
- 不要把当前 `1.31x` Amdahl 数字写成严格 roofline，也不要把 FLA PR 的 H800 结果直接当成 B300 结果。

## 3. 两个人分工

主挑战路线收敛在 C（真实 FlashKDA head-split dispatch 的 FLA 对拍 + FlashKDA 性能比较）。A 路线（`tcgen05` 编译探针）和 B 路线（数值探针）作为支持性证据保留。

### 同学 A：正确性、FLA 对拍、冷启动问题

任务：

- 复用 `tests/test_fwd.py` 的 FLA FP64 `fused_recurrent_kda`，新增或修改 `FlashKDA/tests/` 下的对拍脚本；
- 完成 fixed/varlen、`T=15/16/17/33`、`H=12/96`、initial/final state、continuation case；
- 加入 `torch.isfinite` 检查、误差阈值 `out<=0.005` / `final_state<=0.006`；
- 定位 `parts=2` 首次调用 `cold_diff=4.359e-2` 的原因，并把 warm/cold 拆开统计；
- 多 seed 重复，保存完整原始输出；
- 更新 `challenge_k2.md` 的 correctness 表。

交付标准：

- 官方 FlashKDA baseline 与 FLA reference 对拍通过；
- C 路线的 warm/cold 结果均有明确分类；
- `out` 和 `final_state` 均有误差数字，不只报告 `max_abs_diff`；
- 输出可由单条命令复现。

### 同学 B：性能统计、题面闭环和最终答辩证据

任务：

- 按已确认的题面要求，完善 `bench_flashkda_parallel.py` 的多进程统计；
- 报告 median、p5/p95、标准差，区分 cold-start / warm-start；
- 分开报告 alloc-inclusive 时间与预分配 workspace 的 dispatch/kernel 时间；
- 用 Nsight Systems 检查 streams 是否真实重叠；
- 保存 CUDA、PyTorch、FlashKDA commit、GPU、时钟和命令环境；
- 最终更新 `challenge_k2.md`、`PLAN.md`、`SUMMARY.md` 和 `challenge_a_sm100/README.md`（这些文档中所有本次 C1 文件都位于子模块 `HWs/assignments` 下）；
- 只有在时间充足或老师额外要求时，才选择 K2 的最小 kernel-level candidate。

交付标准：

- baseline 与 split 的多进程统计表；
- 如果追加 kernel variant：有 correctness、SASS、NCU 和端到端性能四类证据；
- 如果不追加：保存当前 dispatch 实验的边界，不把 probe 冒充 FlashKDA 移植成功；
- 性能报告至少有 median 和 p95；
- 结论明确区分“SM100 ISA 可用”“dispatch split 无收益”和“persistent/2-CTA 尚未验证”。

### 两人共同最终检查

1. 所有结论都能指向一个脚本、原始输出或报告表格。
2. 不把 FLA `chunk_kda` 实验冒充 FlashKDA kernel 实验。
3. 不把 `tcgen05` 编译成功冒充 FlashKDA SM100 移植成功。
4. 不把负 speedup 写成实验失败；它是“该改法不值得继续”的结果。
5. 最终结论限定为：**在当前 FlashKDA 实现、B300 硬件和已测试形状下，单纯替换 MMA、增大 BF16 CHUNK 或在 host 侧拆分 head 都没有足够收益；完整 TMA/warp-specialized SM100 后端仍是另一项更大规模工作。**

## 4. 答辩时可能被问到的关键点

| 问题 | 答 |
|---|---|
| 为什么不用真实 K3 权重？ | K3 是 1T 参数模型；只取 `A_log` (96 float) 太小不值得下载；按 K3 §2.1.1 公式合成输入，等价于“参数化方法论”的实证 |
| 为什么 K2 不是 compute-bound？ | 47.5% peak 接近 compute bound 区域，但 30.3% tensor 活跃 + 60% FMA/LSU 表明 GPU 在空等串行状态；真 compute-bound 需要 tensor pipe ≥80% |
| FLA#1075 没用 SM100 吗？ | FLA#1075 是 Hopper (SM90) 改造，不是 SM100；它的 1.4× 在 H800 上；我们 1.31× 是 B300 上，两者同一物理原因 |
| CHUNK=8 行不行？ | 数值更安全，但 GEMM `m8n8k4` / `m16n8k16` 浪费；且 K3 论文没考虑 |
| 为什么不直接换 SM100 tcgen05 试试？ | 我们用 B300 (`sm_103`)，硬件支持 tcgen05，但实测张量核不饱和；改 ISA 救不回 30% 上限外的 70% 串行延迟 |
| 挑战没拿到正收益，作业算完成吗？ | 完成正确性对拍和性能比较后算；负 speedup 可接受，但编译探针或数值探针不能替代这两项 |
| 现有证据已经覆盖 persistent/2-CTA 吗？ | 没有；C 路线只证明 host-side dispatch head split 无收益，persistent/2-CTA 尚未被实现或测试 |

## 5. 当前 C1 文件清单（仅供阅读参考）

复现与测量：

- `HWs/assignments/assignment02/team/c1_flashkda/repro_B300/REPRO.md`
- `HWs/assignments/assignment02/team/c1_flashkda/repro_B300/raw/sass_dump.txt`
- `HWs/assignments/assignment02/team/c1_flashkda/repro_B300/raw/b2_tensor_metrics.csv`

分析与讨论点：

- `HWs/assignments/assignment02/team/c1_flashkda/analysis.md`（6 个讨论点结论+证据）

挑战路线：

- 路线 A：`HWs/assignments/assignment02/team/c1_flashkda/challenge_a_sm100/`（`tcgen05_mma_probe.cu`、`tcgen05_sass_sm100a.txt`、`tcgen05_sass_sm103a.txt`）
- 路线 B：`HWs/assignments/assignment02/team/c1_flashkda/challenge_b1.md`、`challenge_b1/b1_probe.py`、`challenge_b1/upper_bound.py`、`challenge_b1/chunk32_rescale.py` + 各自输出
- 路线 C：`HWs/assignments/assignment02/team/c1_flashkda/challenge_k2.md`、`challenge_k2/bench_chunk_size.py`、`challenge_k2/bench_persistent.py`、`challenge_k2/bench_flashkda_parallel.py` + 输出

计划与答辩：

- `HWs/assignments/assignment02/team/c1_flashkda/PLAN.md`
- `HWs/assignments/assignment02/team/c1_flashkda/SUMMARY.md`
- `HWs/assignments/assignment02/team/c1_flashkda/TASK.md`

## 6. 主要技术参考

- K3 §2.1.1：`g = -5 · σ(e^{A_h} z)`，CHUNK 内 `|cumsum(g)| ≤ 5 · BT`，
  CHUNK=16 时 `reciprocal ≤ e^{80} ≈ 5.5e34` 落在 bf16 范围内（arXiv 2607.24653）
- 官方 FlashKDA deep-dive：CHUNK=16 / bf16 state / 16x16 Neumann 求逆 设计论证
  - <https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md>
- 官方 `tests/test_fwd.py`：用 FLA FP64 `fused_recurrent_kda` 做 oracle
  - <https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea/tests/test_fwd.py>
- NVIDIA CUTLASS tcgen05 文档：`CtaGroup.ONE` 仅接受 `Mma-M ∈ {64, 128}`
  - <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html>
- NVIDIA CUDA C++ Best Practices：`Amdahl S = 1 / ((1-P) + P/N)`
  - <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- Nsight Compute Profiling Guide：occupancy、pipe activity、kernel 复测次数
  - <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- FLA PR #1075：TMA + warp-specialized TLE，H800 上 1.36–1.41×（open / needs-verification）
  - <https://github.com/fla-org/flash-linear-attention/pull/1075>
- cuLA PR #124：把现有 SM90-derived FlashKDA 路径作为 SM100 compatibility path，SM103 disabled
  - <https://github.com/inclusionAI/cuLA/pull/124>