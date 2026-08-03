// #include <__clang_cuda_builtin_vars.h>
#include <bits/extc++.h>

__global__ void clac(float* x, float* y, float* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = 2.0 * x[idx] + y[idx];
    }
}

int main(int argc, char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
    }
    int n;
    n = std::atoi(argv[1]);
    std::cerr << n << "\n";
    float* x = (float*)std::malloc(n * sizeof(float));
    float* y = (float*)std::malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        x[i] = ((i % 2048) - 1024) * 0.5f;
        y[i] = ((i % 1024) - 512);
    }
    float* dx;
    float* dy;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, n * sizeof(float));
    float* dz;
    cudaMalloc(&dz, n * sizeof(float));

    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clac<<<blocks, threads>>>(dx, dy, dz, n);

    float* z = (float*)std::malloc(n * sizeof(float));
    cudaMemcpy(z , dz, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    long double sum = 0.;
    for (int i = 0; i < n; i++) {
        sum += z[i];
    }

    std::printf("SUM=%.0LF\n", sum);
    return 0;
}
