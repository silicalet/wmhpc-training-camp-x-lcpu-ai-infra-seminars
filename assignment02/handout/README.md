# handout 构建

原始材料是 `src/assignment02.md`，tex 与 pdf 由它生成，不要手工编辑
生成物。构建:`./build.sh`(依赖 pandoc、XeLaTeX、ctex 与 Fandol 字体)。
版式与 assignment01 的 handout 一致(模板 `template.tex`)。

从仓库根目录将报告输出到实验记录目录：

```bash
BUILD_DIR=../results/report bash assignment02/handout/build.sh
```

Nix 环境可用以下命令准备完整的 TeX 依赖，再执行构建：

```bash
nix build --impure --expr 'let pkgs = import (builtins.getFlake "nixpkgs").outPath {}; in import ./assignment02/handout/texlive.nix { inherit pkgs; }' --out-link /tmp/assignment02-texlive
PATH=/tmp/assignment02-texlive/bin:$PATH BUILD_DIR=../results/report bash assignment02/handout/build.sh
```

脚本先用 Pandoc 生成 TeX，再运行两遍 XeLaTeX；每遍编译日志保存在输出目录。

新报告位于 `assignment02/results/report/assignment02.pdf`；此命令不覆盖
`assignment02/assignment02.pdf` 或 `handout/assignment02.pdf` 中的旧版题本。
高亮代码块支持长 token 换行。压轴题内的 `answer` 渲染为题框之后的独立
可分页作答框；超过 20 行的作答表格移出色框，由 `longtable` 分页并重复表头。

## md 源的约定

普通 markdown(标题、粗体、行内代码、代码块、pipe 表格、列表、$数学$、
raw latex 均可)之外,三种结构标记由 `filters/boxes.lua` 翻译:

题目标题(三级标题 + 属性;`opt`/`file` 可省略):

    ### 4.1 {.prob type=FROM-SCRATCH file=cuda/m4_gemm/01_tiled.cu}
    ### 4.4 {.prob type=FROM-SCRATCH opt=Optional}

参考资料框 / Editor's Note 框 / 作答框:

    ::: reading
    PTX ISA 9.7.14(mma 一节)、课件 S026-S030。
    :::

    ::: lookback
    模块收尾的评注。
    :::

    ::: answer
    实测现象与解释。
    :::

压轴题框(题目整个包进框里,标题自拟):

    ::: {.capstone title="prob 3.2(FROM-SCRATCH):tcgen05 单 tile" file=cuda/m3_tcgen05/02_single_tile.cu}
    题面……
    :::

语法样例见 `src/_syntax_demo.md`,`./build.sh src/_syntax_demo.md`
可单独编译它验证工具链。
