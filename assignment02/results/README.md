# Assignment 02 实验记录

本次实验日期：2026-09-09。主报告在 `../handout/src/assignment02.md`，生成的 TeX 在 `report/assignment02.tex`。本机未安装 xelatex，因此没有重新生成 PDF；仓库原来的生成文件未由本次任务覆盖。

本地为 AMD Ryzen AI 9 HX 370 / CUDA 12.9.86，NVIDIA 驱动不可用。无卡题用本地 CPU 完成，GPU 题通过 `ssh -F ~/.ssh/config b300-login`，在 Slurm job 23618 的一张 B300 上运行。B300 为 sm_103、148 SM，CUDA 13.0.88，编译使用 `ARCH=100f`。独立远端目录是 `~/assignment02-codex-20260909`。

## 判测与证据

| 内容 | 记录 |
|---|---|
| 本地硬件、驱动与工具链 | `local/environment.log` |
| 1.1 / 2.2 / 2.3 host 判测 | `local/host-tests.log` |
| 5.1 outlier 完整数字 | `local/outlier.log` |
| 5.2 三个 pytest | `local/pytest.log` |
| B300 配置、设备查询与占卡检查 | `b300/environment.log`, `device.log`, `gpu-details.csv`, `processes-before.log`, `processes-after.log` |
| 0.1 匹配与不匹配架构 | `b300/01_first_mma.log`, `arch-mismatch.log` |
| DEBUG 原始源码及运行 | `original/`, `b300/debug-before.log` |
| 1.2 修复后 | `b300/02_bug_fragment.log` |
| 1.3 五个 seed | `b300/judge-fp8.log` |
| 1.4 两条装载路径、编译输出与指令计数 | `b300/04_ldmatrix.log`, `ldmatrix.ptx.txt`, `ldmatrix.sass.txt`, `ldmatrix-counts.txt` |
| 1.5 bank conflict | `b300/05_ldsm_stride.log`, `ncu-stride.csv` |
| 3.2 / 3.3 多 seed 与多轮 | `b300/judge-tcgen.log` |
| 3.2 去掉 proxy fence 的实验 | `b300/no-fence.log`（有意错误版本也 PASS，不能据此省略 fence） |
| 3.4 CTA pair | `b300/04_cta_pair.log`, `ncu-cta.csv` |
| 4.1–4.3 / naive 同形状对照 | `b300/01_tiled.log`, `02_tma.log`, `03_pipeline.log`, `naive.log` |
| 4.3 8 项 stage 扫描及实际占用 | `b300/stages.log`, `pipeline-short-k.log`, `ncu-stage-*.csv` |
| 4.5 全部 63 项 thin GEMM | `b300/thin.log` |
| 5.3 编码、quant data/SF、cuBLASLt 消费 | `b300/03a_encode_check.log`, `03b_nvfp4_quant.log`, `test_fp4_gemm.log` |
| 5.3 / 5.4 同形状带宽探针 | `b300/03c_ceiling_probe.log`, `bandwidth.log` |
| 5.4 初版与优化后完整候选配置 | `b300/04_fused_rms_nvfp4.log`, `fused-tuned.log` |
| 5.3 / 5.4 大形状 NCU | `b300/ncu-quant-large.csv`, `ncu-fused-large.csv` |
| 6.1 两种架构的 CUDA、IR、cubin 编译和 SASS | `tilelang/`, `b300/tilelang.log` |

`.cubin`、本地 binary 和虚拟环境被 gitignore 排除，源码/IR/SASS 文本和构建日志保留。带 `ncu-` 的输出中，程序自身打印的时间受 profiler 扰动，不用于报告的性能表。占用 API 对 tcgen05 kernel 返回的 1 与实测 active warp 不一致，报告同时列出 shared 容量上界与 NCU，未将该 API 数字当作实际驻留量。

## 复现

在本地 CPU 环境运行：

```bash
cd assignment02
uv venv .venv
uv pip install --python .venv/bin/python pytest
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest tests/ -q
.venv/bin/python kernels/quant_outlier.py
ARCH=89 make -B -C cuda run/m1_sm80/01_fragment_map run/m2_smem/02_descriptor run/m2_smem/03_swizzle
```

GPU 编译和测量应在 Slurm 分配的 B300 计算节点执行，登录节点只用于传输和提交任务：

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd ~/assignment02-codex-20260909
bash experiments/run_required.sh
```

脚本强制重编，GPU 程序带 `timeout -k 5`，默认把新结果写入 `results/recheck/`，不覆盖本次证据。性能会随频率和调度发生波动。

补充实验：

```bash
cd cuda
nvcc -O2 -std=c++17 -gencode arch=compute_100f,code=sm_100f ../experiments/bandwidth.cu -o bin/bandwidth
./bin/bandwidth
nvcc -O2 -std=c++17 -gencode arch=compute_100f,code=sm_100f ../experiments/naive_matmul.cu -lcublas -o bin/naive
./bin/naive
ncu --csv --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bin/m1_sm80/05_ldsm_stride
ncu --csv --launch-count 2 --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./bin/m3_tcgen05/04_cta_pair
ncu --csv --kernel-name regex:nvfp4_quant_kernel --launch-count 1 --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./bin/bandwidth 4096 7168
ncu --csv --kernel-name regex:rms_quant --launch-count 1 --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active ./bin/bandwidth 4096 7168
```

6.1 使用固定版本，脚本只编译，不执行 sm_90a kernel：

```bash
uv pip install --python .venv/bin/python tilelang==0.1.13
.venv/bin/python experiments/tilelang_lower.py
```

生成报告（从仓库根目录）：

```bash
pandoc assignment02/handout/src/assignment02.md \
  --from markdown+fenced_divs+pipe_tables+raw_tex \
  --template assignment02/handout/template.tex \
  --lua-filter assignment02/handout/filters/boxes.lua \
  --highlight-style=pygments \
  -o assignment02/results/report/assignment02.tex
```

范围：必做 M0–M6。额外 4.4、5.3(d) 和团队 C1/C2 未选做；5.4 的向量装载优化已完成。已有 `plan.md` 的团队进展声明不作为本次验证证据。
