#!/usr/bin/env bash
# handout 构建:md 源 → tex → pdf。
# 用法:BUILD_DIR=../results/report ./build.sh [src/assignment02.md]
# 依赖:pandoc(>=3)、XeLaTeX、ctex 与 Fandol 字体。
set -euo pipefail
cd "$(dirname "$0")"
SRC=${1:-src/assignment02.md}
BASE=$(basename "$SRC" .md)
BUILD_DIR=${BUILD_DIR:-.}
mkdir -p "$BUILD_DIR"
pandoc "$SRC" \
    --from markdown+fenced_divs+pipe_tables+raw_tex \
    --template template.tex \
    --lua-filter filters/boxes.lua \
    --highlight-style=pygments \
    -o "$BUILD_DIR/$BASE.tex"
for pass in 1 2; do
    if ! xelatex -halt-on-error -interaction=nonstopmode \
        -output-directory="$BUILD_DIR" "$BUILD_DIR/$BASE.tex" \
        > "$BUILD_DIR/$BASE-pass$pass.log" 2>&1; then
        tail -40 "$BUILD_DIR/$BASE-pass$pass.log"
        exit 1
    fi
done
echo "生成 $BUILD_DIR/$BASE.pdf"
