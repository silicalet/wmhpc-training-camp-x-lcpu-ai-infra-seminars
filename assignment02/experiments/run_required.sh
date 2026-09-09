#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
out=${RESULTS_DIR:-$root/results/recheck}
mkdir -p "$out"
export ARCH=${ARCH:-100f}
run() {
    local name=$1
    shift
    printf '%q ' "$@" > "$out/$name.log"
    printf '\n' >> "$out/$name.log"
    "$@" 2>&1 | tee -a "$out/$name.log"
}
run environment nvidia-smi
run nvcc nvcc --version
cd cuda
for f in m0_env/01_first_mma m1_sm80/01_fragment_map m1_sm80/02_bug_fragment \
    m1_sm80/03_mma_fp8 m1_sm80/04_ldmatrix m1_sm80/05_ldsm_stride \
    m2_smem/02_descriptor m2_smem/03_swizzle m3_tcgen05/02_single_tile \
    m3_tcgen05/03_bug_mbarrier m3_tcgen05/04_cta_pair m4_gemm/01_tiled \
    m4_gemm/02_tma m4_gemm/03_pipeline m4_gemm/05_thin_gemm \
    m5_lowprec/03a_encode_check m5_lowprec/03b_nvfp4_quant \
    m5_lowprec/test_fp4_gemm m5_lowprec/03c_ceiling_probe m5_lowprec/04_fused_rms_nvfp4; do
    name=${f//\//-}
    run "build-$name" make -B "bin/$f"
    if [[ $f == m4_gemm/05_thin_gemm ]]; then
        run "$name" timeout -k 5 240 "./bin/$f" 2250 8000
    else
        run "$name" timeout -k 5 240 "./bin/$f"
    fi
done
(cd m1_sm80; run judge-fp8 bash judge_mma_fp8.sh 03_mma_fp8.cu)
(cd m3_tcgen05; run judge-tile bash judge_tile.sh; run judge-mbar bash judge_mbar.sh)
run stages bash m4_gemm/sweep_stages.sh
# sweep 的最终二进制为 S=6；恢复默认 S=3。
make -B bin/m4_gemm/03_pipeline
cd "$root"
if [[ -x .venv/bin/python ]]; then
    run python-tests .venv/bin/python -m pytest tests/ -q
    run outlier .venv/bin/python kernels/quant_outlier.py
    run tilelang .venv/bin/python experiments/tilelang_lower.py
fi
