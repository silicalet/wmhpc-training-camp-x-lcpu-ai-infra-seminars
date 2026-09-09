// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

#ifndef STAGES
#define STAGES 3
#endif

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NSTAGE = STAGES;

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

// 非阻塞版:成功返回 true。机会式深预取用它。
__device__ inline bool mbar_try(uint32_t mbar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(mbar), "r"(phase));
    return done;
}


static CUtensorMap tensor_map(void *ptr, int rows, int K, int tile_rows) {
    CUtensorMap map{};
    uint64_t dims[2] = {uint64_t(K), uint64_t(rows)};
    uint64_t strides[1] = {uint64_t(K) * 2};
    uint32_t box[2] = {64, uint32_t(tile_rows)}, elem[2] = {1, 1};
    CUresult e = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, ptr, dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (e != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", int(e));
        std::exit(1);
    }
    return map;
}

__device__ inline void load_tma(uint8_t *sa, uint8_t *sb, uint32_t full,
                                const CUtensorMap *a, const CUtensorMap *b,
                                int k, int m, int n) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24576;"
        :: "r"(full) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(uint32_t(__cvta_generic_to_shared(sa))), "l"(a), "r"(k), "r"(m), "r"(full) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(uint32_t(__cvta_generic_to_shared(sb))), "l"(b), "r"(k), "r"(n), "r"(full) : "memory");
}

__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    __shared__ __align__(8) uint64_t bar;
    __shared__ uint32_t addr;
    int t = threadIdx.x, warp = t >> 5;
    uint32_t mb = __cvta_generic_to_shared(&bar);
    if (t == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        uint32_t p = __cvta_generic_to_shared(&addr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" :: "r"(p) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    uint32_t ta = addr;
    constexpr int S = NSTAGE;
    static_assert(S > 0);
    __shared__ __align__(8) uint64_t full[S], empty[S];
    int tm = blockIdx.x * 128, tn = blockIdx.y * 64, count = K / 64;
    if (t == 0) {
        for (int i = 0; i < S; i++) {
            uint32_t f = __cvta_generic_to_shared(full + i);
            uint32_t e = __cvta_generic_to_shared(empty + i);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(f) : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(e) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) {
        int next = 0;
        for (; next < S && next < count; next++) {
            auto *sa = smem + next * 24576;
            load_tma(sa, sa + 16384, __cvta_generic_to_shared(full + next),
                &tmapA, &tmapB, next * 64, tm, tn);
        }
        for (int it = 0; it < count; it++) {
            // 当前轮必须已发射；机会式预取失败后在这里阻塞补发。
            if (next == it) {
                int q = next % S;
                mbar_wait(__cvta_generic_to_shared(empty + q), ((next / S) - 1) & 1);
                auto *sa = smem + q * 24576;
                load_tma(sa, sa + 16384, __cvta_generic_to_shared(full + q),
                    &tmapA, &tmapB, next * 64, tm, tn);
                next++;
            }
            while (next < count && next < it + S) {
                int q = next % S;
                if (!mbar_try(__cvta_generic_to_shared(empty + q), ((next / S) - 1) & 1)) break;
                auto *sa = smem + q * 24576;
                load_tma(sa, sa + 16384, __cvta_generic_to_shared(full + q),
                    &tmapA, &tmapB, next * 64, tm, tn);
                next++;
            }
            int q = it % S;
            mbar_wait(__cvta_generic_to_shared(full + q), (it / S) & 1);
            auto *sa = smem + q * 24576;
            auto *sb = sa + 16384;
            mb = __cvta_generic_to_shared(empty + q);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for (int k = 0; k < 64; k += 16) {
            uint64_t da = make_desc_sm100(__cvta_generic_to_shared(sa) + k * 2, 0, 1024, 2);
            uint64_t db = make_desc_sm100(__cvta_generic_to_shared(sb) + k * 2, 0, 1024, 2);
            uint32_t id = (1u << 4) | (1u << 7) | (1u << 10) | (8u << 17) | (8u << 24);
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
                :: "r"(ta), "l"(da), "l"(db), "r"(id), "r"(int(it != 0 || k != 0)) : "memory");
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :: "r"(mb) : "memory");
        }
        mbar_wait(__cvta_generic_to_shared(empty + (count - 1) % S), ((count - 1) / S) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for (int c = 0; c < 64; c += 8) {
        float v[8];
        uint32_t p = ta + ((warp * 32) << 16) + c;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
              "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "r"(p) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i = 0; i < 8; i++) gD[(size_t(tm) + t) * N + tn + c + i] = v[i];
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(ta) : "memory");
    }

}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M <= 0 || N <= 0 || K <= 0 || M % BM || N % BN || K % BK) {
        printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    // 与 4.2 相同的 tensor map。
    CUtensorMap tmapA = tensor_map(dA, M, K, BM);
    CUtensorMap tmapB = tensor_map(dB, N, K, BN);

    dim3 grid(M / BM, N / BN);
    // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
    size_t smemBytes = (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    int blocks;
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, gemm_pipeline));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gemm_pipeline, 128, smemBytes));
    printf("RESOURCES S=%d dynamic_smem=%zu static_smem=%zu registers=%d blocks/SM=%d\n",
        NSTAGE, smemBytes, attr.sharedSizeBytes, attr.numRegs, blocks);
    auto launch = [&] {
        gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA,
                                                tmapB);
    };
    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF,
                 K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; i++) bad += got[i] != ref[i];

    int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
           "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
