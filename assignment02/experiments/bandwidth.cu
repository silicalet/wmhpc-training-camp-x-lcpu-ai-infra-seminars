#define main fused_judge_main
#include "../cuda/m5_lowprec/04_fused_rms_nvfp4.cu"
#undef main
#define main probe_original_main
#include "../cuda/m5_lowprec/03c_ceiling_probe.cu"
#undef main
int main(int argc, char **argv) {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    for (auto shape : {std::pair{1,4096}, {16,4096}, {256,4096}, {1024,4096},
         {4096,4096}, {16384,4096}, {4096,7168}, {16384,7168}, {4096,8192}, {16384,8192}}) {
        int m = shape.first, k = shape.second;
        if (argc == 3 && (m != atoi(argv[1]) || k != atoi(argv[2]))) continue;
        size_t n = size_t(m) * k;
        __nv_bfloat16 *x, *w;
        uint8_t *d, *s;
        CUDA_CHECK(cudaMalloc(&x, n * 2));
        CUDA_CHECK(cudaMalloc(&w, k * 2));
        CUDA_CHECK(cudaMalloc(&d, n / 2));
        CUDA_CHECK(cudaMalloc(&s, nvfp4_sf_bytes(m, k)));
        CUDA_CHECK(cudaMemset(x, 0x3c, n * 2));
        CUDA_CHECK(cudaMemset(w, 0x3c, k * 2));
        CUDA_CHECK(cudaMemset(s, 0, nvfp4_sf_bytes(m, k)));
        float tp = time_avg_ms([&] { launch_probe(x,d,s,m,k,sms); }, 100);
        float tq = time_avg_ms([&] { launch_nvfp4_quant(x,d,s,m,k,sms); }, 100);
        float tf = time_avg_ms([&] { launch_fused(x,w,d,s,m,k,1e-6f,sms); }, 100);
        double bytes = n * 2.5625;
        printf("M=%d K=%d probe_us=%.3f quant_us=%.3f fused_us=%.3f probe_GBs=%.1f quant_GBs=%.1f fused_GBs=%.1f quant/ceiling=%.4f fused/ceiling=%.4f\n",
            m,k,tp*1000,tq*1000,tf*1000,effective_gbps(bytes,tp),effective_gbps(bytes,tq),effective_gbps(bytes,tf),tp/tq,tp/tf);
        cudaFree(x); cudaFree(w); cudaFree(d); cudaFree(s);
    }
}
