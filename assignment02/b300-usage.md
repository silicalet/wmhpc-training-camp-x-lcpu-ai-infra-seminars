# B300 服务器登录与使用指南

本文记录 `b300` 服务器的登录、文件传输、Slurm 资源申请和 Python 环境使用方法。

## 1. 服务器概况

本机 SSH 配置中定义了两个别名：

- `b300-jump`：公网跳板机；通常不需要手动登录。
- `b300-login`：日常使用的登录节点；SSH 会自动经过跳板机。

连接链路如下：

```text
本机
  └─ b300-jump
       └─ b300-login
            └─ Slurm 计算节点 dev-slurm
```

已确认的账户与环境：

| 项目 | 值 |
|---|---|
| 用户名 | `99043118` |
| 用户主目录 | `/home/lcpu/99043118` |
| 登录节点 | `b300-login` |
| 计算节点 | `dev-slurm` |
| 作业调度器 | Slurm 23.11.4 |
| Python | Python 3.12.3 |
| Python 包管理工具 | uv 0.12.1 |

计算节点资源：

- 240 CPU
- 约 2 TB 内存
- 8 张 NVIDIA B300 SXM6 AC
- 每张 GPU 显存约 275040 MiB（约 268.6 GiB）
- NVIDIA 驱动版本 580.126.09

## 2. 登录服务器

直接使用 SSH 别名：

```bash
ssh b300-login
```

SSH 已配置 `ProxyJump`，不需要先手动登录跳板机。

登录后可以确认当前环境：

```bash
hostname
id -un
pwd
```

预期位于登录节点，主目录为：

```text
/home/lcpu/99043118
```

> 登录节点只适合传输文件、编辑代码和提交作业。训练、编译或其他高负载任务应通过 Slurm 在计算节点上运行。

## 3. 文件传输

以下命令均在本机执行。

### 3.1 上传单个文件

```bash
scp local-file b300-login:~/
```

### 3.2 上传目录

```bash
scp -r local-directory b300-login:~/
```

### 3.3 使用 rsync 上传项目

推荐使用 `rsync`，它支持增量同步和断点续传：

```bash
rsync -avP ./project/ b300-login:~/project/
```

注意源目录末尾的 `/`：它表示同步目录中的内容，而不是再创建一层 `project` 目录。

### 3.4 从服务器下载结果

```bash
rsync -avP b300-login:~/project/results/ ./results/
```

## 4. 查看 Slurm 资源

登录服务器后，查看分区和节点状态：

```bash
sinfo
```

查看自己的任务：

```bash
squeue -u "$USER"
```

当前提供两个分区：

| 分区 | 最长运行时间 | 用途 |
|---|---:|---|
| `gpu` | 1 小时 | GPU 任务；默认分区 |
| `cpu` | 8 小时 | CPU 任务；单任务最多 64 CPU |

GPU 分区中，每申请一张 GPU，Slurm 默认同时分配约 30 CPU 和 240000 MB 内存。最好仍在命令中显式声明实际需要的 CPU 和内存。

## 5. 交互式 GPU 作业

申请一张 GPU，并进入计算节点的交互式 Shell：

```bash
srun \
  --partition=gpu \
  --gres=gpu:1 \
  --time=01:00:00 \
  --pty bash -l
```

进入后检查节点和 GPU：

```bash
hostname
nvidia-smi
```

此时 `hostname` 应显示：

```text
dev-slurm
```

显式限制 CPU 和内存的示例：

```bash
srun \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=01:00:00 \
  --pty bash -l
```

申请多张 GPU：

```bash
srun \
  --partition=gpu \
  --gres=gpu:4 \
  --time=01:00:00 \
  --pty bash -l
```

使用完成后退出 Shell，Slurm 会释放资源：

```bash
exit
```

> `gpu` 分区最长运行时间为 1 小时，不能在该分区提交更长的任务。

## 6. 交互式 CPU 作业

例如申请 8 个 CPU、32 GiB 内存，最长运行 2 小时：

```bash
srun \
  --partition=cpu \
  --cpus-per-task=8 \
  --mem=32G \
  --time=02:00:00 \
  --pty bash -l
```

确认已经进入计算节点：

```bash
hostname
```

使用完毕后执行：

```bash
exit
```

## 7. 提交后台 GPU 作业

长时间运行或不需要交互的任务应使用 `sbatch`。

创建 `train.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/project"
source "$HOME/venvs/project/bin/activate"

hostname
nvidia-smi
python3 train.py
```

创建日志目录并提交：

```bash
mkdir -p logs
sbatch train.sbatch
```

提交成功后，`sbatch` 会输出任务 ID，例如：

```text
Submitted batch job 1234
```

查看任务状态：

```bash
squeue -u "$USER"
```

实时查看标准输出：

```bash
tail -f logs/train-1234.out
```

取消任务：

```bash
scancel 1234
```

查看当天已结束的任务：

```bash
sacct -u "$USER" --starttime today
```

## 8. Python 环境

服务器提供 Python 3.12 和 `uv`，未发现系统级 Conda。推荐为每个项目创建独立虚拟环境。

创建环境：

```bash
mkdir -p ~/venvs
uv venv ~/venvs/project
```

激活环境：

```bash
source ~/venvs/project/bin/activate
```

安装依赖：

```bash
uv pip install numpy
```

如果项目包含 `pyproject.toml`：

```bash
cd ~/project
uv sync
uv run python train.py
```

登录节点和计算节点共享用户主目录，因此在登录节点创建的虚拟环境可以在 Slurm 作业中直接使用。

## 9. 推荐工作流程

在本机上传代码：

```bash
rsync -avP ./project/ b300-login:~/project/
```

登录服务器：

```bash
ssh b300-login
```

进入项目并申请一张 B300：

```bash
cd ~/project
srun -p gpu --gres=gpu:1 --time=01:00:00 --pty bash -l
```

确认环境并运行程序：

```bash
hostname
nvidia-smi
uv run python train.py
```

完成后释放资源：

```bash
exit
```

最后在本机下载结果：

```bash
rsync -avP b300-login:~/project/results/ ./results/
```

## 10. 常用命令速查

```bash
# 登录
ssh b300-login

# 查看集群资源
sinfo

# 查看自己的任务
squeue -u "$USER"

# 申请交互式 GPU
srun -p gpu --gres=gpu:1 -t 01:00:00 --pty bash -l

# 申请交互式 CPU
srun -p cpu -c 8 --mem=32G -t 02:00:00 --pty bash -l

# 提交后台任务
sbatch train.sbatch

# 取消任务
scancel JOB_ID

# 查看历史任务
sacct -u "$USER" --starttime today

# 查看 GPU
nvidia-smi

# 退出计算节点并释放资源
exit
```

## 11. 常见问题

### 任务一直显示 PENDING

检查等待原因：

```bash
squeue -u "$USER" -o "%.18i %.9P %.16j %.8T %.10M %.6D %R"
```

常见原因包括 GPU 已被占用、请求的资源过多或请求时间超过分区上限。

### GPU 程序在登录节点看不到 GPU

这是正常现象。需要先通过 `srun` 或 `sbatch` 申请 GPU，再在 `dev-slurm` 上运行程序。

### SSH 连接失败

先显示详细连接日志：

```bash
ssh -v b300-login
```

配置使用跳板机，因此需要确认本机到跳板机的网络畅通，并确认 SSH 密钥仍然有效。

### 作业结束后找不到输出

`#SBATCH --output` 和 `#SBATCH --error` 的相对路径基于提交 `sbatch` 时所在的目录。提交前创建日志目录：

```bash
mkdir -p logs
```
