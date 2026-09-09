#include <cublas_v2.h>
#include <random>
#include "../cuda/common.h"
constexpr int BM=128, BN=64, BK=64;
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
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
    std::vector<float> hA(nA), hB(nB);
    for (auto& v : hA) v = (float)dist(rng);
    for (auto& v : hB) v = (float)dist(rng);
    float *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 4));
    CUDA_CHECK(cudaMalloc(&dB, nB * 4));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 4, cudaMemcpyHostToDevice));
    // 先填 NaN 模式:kernel 没写满/没写对时判测必 FAIL,不受残留数据干扰
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    auto launch = [&] {
        matmul_naive<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(dA, dB, dD, M, N, K);
    };
    launch();
    CUDA_CHECK_KERNEL();

    // cuBLAS 参考(bf16 输入 f32 累加;小整数下与 tensor core 逐位一致)
    // D[M,N] 行主序 = D^T 列主序:C_col[N,M] = B_col[K,N]^T x A_col[K,M]
    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_32F,
                 N, dA, CUDA_R_32F, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
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
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_32F, N, dA, CUDA_R_32F, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F_PEDANTIC,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[naive fp32] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f TFLOPS  "
           "(cuBLAS %.1f, 达成率 %.0f%%)\n",
           M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
           100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
