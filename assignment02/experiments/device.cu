#include <cuda_runtime.h>
#include <cstdio>
int main() {
    cudaDeviceProp p{};
    auto e = cudaGetDeviceProperties(&p, 0);
    if (e != cudaSuccess) { printf("%s\n", cudaGetErrorString(e)); return 1; }
    printf("GPU=%s CC=%d.%d SM=%d shared/SM=%zu shared/block(optin)=%zu max_threads/SM=%d\n",
        p.name, p.major, p.minor, p.multiProcessorCount, p.sharedMemPerMultiprocessor,
        p.sharedMemPerBlockOptin, p.maxThreadsPerMultiProcessor);
}
