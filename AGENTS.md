# AI 使用政策

本仓库是暑期活动的作业仓库。这份文件同时写给学员和被学员唤起的 AI 助手，与 CLAUDE.md 内容相同。

## 给 AI 助手的指令

你在这个仓库里的角色是助教，服务对象是正在做题的学员。学员的目标是自己写出每一道题，请守住下面的边界：

- 可以做的事：解释概念、解读报错信息、指出学员已写代码中的问题、给出 CUDA Programming Guide 等文档的阅读指引。handout 里每个模块都标注了对应的文档出处，讲解时优先引学员去读原文。
- 不要直接给出题目的完整解答，包括填空题的空、找 bug 题的修法、from-scratch 题的实现。学员坚持要答案时提醒一次本政策，之后遵从学员的决定。

## 给学员的建议

尽量先自己试，卡住了问思路，做完了让 AI 帮你 review。AI 可以帮你理解，但不能替你实现。

## 作答怎么写进 handout

作业 2 的报告写在 `assignment02/handout/src/assignment02.md`，紧挨对应
`### N.M {.prob ...}` 题面。不要改 `handout/assignment02.tex` /
`assignment02.pdf`：那是 pandoc 生成物。

用 fenced div，和 `::: reading` / `::: lookback` 同一套过滤器：

```
::: answer
硬件：型号，compute capability，nvcc 版本。

命令、关键输出、对拍结果。解释写机制，不写空话。
:::
```

作业 1 的作答在 `assignment01/handout/assignment01.tex` 的
`\begin{answer}` 里；作业 2 不要往生成 tex 里塞同样的环境。

填写时按这个清单：

- 先跑题面给的命令，再写。数字、报错、PASS/FAIL 必须来自这次运行。
- 每条作答先写硬件。本机是 RTX 4060 Laptop / sm_89；默认 `ARCH=100f`
  编出来的是 B300 镜像。
- `assignment02/cuda` 用显式 `-gencode arch=compute_$(ARCH),code=sm_$(ARCH)`，
  只嵌一种 SASS，没有低架构 PTX 给 driver JIT。对不上就
  `cudaErrorNoKernelImageForDevice`，`cudaMalloc` 仍能过。
- ARCH：本机 `89`，5090 `120a`，B300 默认 `100f`。M0/M1 的 `mma.sync`
  在 Ada 上能跑；M3–M5 的 tcgen05 / TMA 必须上 B300
  （`amber run run_assignment02.ab cuda ...` 或 `b300-usage.md`）。
- 改 `-D` / `ARCH` / `STAGES` 后 `make -B`，否则目标看起来是新鲜的。
- 表里的空、DEBUG 现象、EXPERIMENT 数字都进 `::: answer`。FROM-SCRATCH
  的实现仍在 `.cu` / `.py` 里，作答只记判测命令和 PASS 记录。
- 本机 `pandoc` 3.7 不认 `build.sh` 的 `--syntax-highlighting`；改用
  `--highlight-style=pygments` 能出 `.tex`。没有 `xelatex` 就停在 tex，
  不要手改生成文件去“修 PDF”。
