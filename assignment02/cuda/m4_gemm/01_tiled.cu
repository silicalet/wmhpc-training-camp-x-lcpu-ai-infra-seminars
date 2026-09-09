// 问题 4.1(FROM-SCRATCH):tiling——把 3.2 的单 tile 扩成完整 GEMM。
//
// 形状默认 4096^3,bf16 输入,f32 累加,cta_group::1。tile 固定
// BM=128 BN=64 BK=64,每 block 128 线程,grid = (M/BM, N/BN)。
// 判测(main 已给出)对 cuBLAS 逐位严格相等:数据是小整数,和的
// 绝对值 <= 2^24,f32 累加无舍入,顺序无关,可以逐位判。
//
// 相对 3.2 的新内容只有两件:
//   - 每个 block 按 blockIdx 认领输出 tile(tileM, tileN)
//   - K 维不再一次装完:iters = K/BK 轮,每轮 staging 一段、发一组
//     mma 累加到同一块 TMEM,循环结束再走 epilogue
// TMEM 分配、descriptor、swizzled staging、epilogue 都是 3.2 已有的,
// 原样搬过来改数字即可。
//
// 交付:PASS + 本文件输出的 TFLOPS/达成率(梯子表第一行,后面 4.2/4.3
// 各补一行);回答 handout 4.1 的问题(达成率 vs 机器平衡点,此时瓶颈
// 在哪个环节)。
//
// 运行:make run/m4_gemm/01_tiled;自定形状 ./bin/m4_gemm/01_tiled M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

constexpr int BM = 128, BN = 64, BK = 64;

// 128B swizzle 的物理偏移(2.3/3.2 用过的同一个)。
__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

// SM100 smem descriptor(2.2 的位域)。
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

__global__ void gemm_tiled(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                           float* gD, int M, int N, int K) {
    // smem 用动态分配(main 已按 (BM+BN)*BK*2 + 1024 传入),基址对齐
    // 到 1024(swizzle atom 的要求);A 区 [0, BM*BK*2),B 区随后。
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
    auto *sa = smem;
    auto *sb = smem + 128 * 64 * 2;
    int tm = blockIdx.x * 128, tn = blockIdx.y * 64;
    for (int it = 0; it < K / 64; it++) {
        for (int i = t; i < 128 * 64; i += 128) {
            int r = i / 64, k = i % 64;
            *reinterpret_cast<__nv_bfloat16 *>(sa + swz128(r, k * 2)) = gA[(size_t(tm) + r) * K + it * 64 + k];
        }
        for (int i = t; i < 64 * 64; i += 128) {
            int r = i / 64, k = i % 64;
            *reinterpret_cast<__nv_bfloat16 *>(sb + swz128(r, k * 2)) = gB[(size_t(tn) + r) * K + it * 64 + k];
        }
#ifndef OMIT_PROXY_FENCE
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
        __syncthreads();
        if (t == 0) {
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
        mbar_wait(mb, it & 1);
        __syncthreads();  // 所有线程看到消费完成后才允许覆盖 shared memory。
    }
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
    // 先填 NaN 模式:kernel 没写满/没写对时判测必 FAIL,不受残留数据干扰
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    dim3 grid(M / BM, N / BN);
    size_t smemBytes = (size_t)(BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_tiled,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_tiled<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K);
    };
    launch();
    CUDA_CHECK_KERNEL();

    // cuBLAS 参考(bf16 输入 f32 累加;小整数下与 tensor core 逐位一致)
    // D[M,N] 行主序 = D^T 列主序:C_col[N,M] = B_col[K,N]^T x A_col[K,M]
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
    printf("[4.1 tiled] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f TFLOPS  "
           "(cuBLAS %.1f, 达成率 %.0f%%)\n",
           M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
           100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
