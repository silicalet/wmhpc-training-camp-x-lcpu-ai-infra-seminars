# Weiming HPC Training Camp x LCPU AI Infra Seminars

北京大学未名超算队（The Radiance of Weiming）与北京大学学生 Linux 俱乐部（LCPU）合办的暑期 AI Infra 系列活动仓库。每次 session 的作业与配套代码放在对应的 `assignmentXX/` 目录。

[official website](https://infra.seminars.lcpu.dev)

## 内容

系列从 GPU 编程起步，此后依次进入 memory hierarchy、Tensor Core、软件流水、DSL 与编译器、通信、推理系统、RL 系统。完整时间表以活动推送为准。

| 目录 | Session | 主题 |
| --- | --- | --- |
| [assignment01](assignment01/) | 1（7.26） | GPU & GPU Programming——SIMD/SIMT、CUDA 编程模型、Triton/TileLang |

## 使用方式

每个 assignment 目录自带 README、handout PDF 与基础代码。

### Nix devShell

仓库提供基于 flake-parts 的开发环境，包含 CUDA Toolkit、Python、uv 和常用编译工具：

```bash
nix develop
cd assignment01
uv sync --extra tilelang
```

Python 依赖（包括 Torch、Triton 和可选的 TileLang）由各 assignment 的
`pyproject.toml` 和 uv 管理，不会打包进 Nix devShell。首次进入环境和首次
`uv sync` 都需要联网；运行 CUDA 程序仍需要宿主机安装兼容的 NVIDIA 驱动。

## 关于 AI 使用

policy 见 [CLAUDE.md](CLAUDE.md)（[AGENTS.md](AGENTS.md) 内容相同，供其他工具读取）。AI 可以帮你理解，但不能替你实现。
