// 问题 5.3(b):实现 NVFP4 quant kernel。
//
// 输入 bf16 矩阵 [M, K](K 是 16 的倍数),输出:
//   dataOut:e2m1 打包数据,每行 K/2 byte,低 nibble 放偶数下标元素
//   sfOut:  e4m3 SF,swizzled 布局(偏移用 nvfp4_common.h 的
//           sf_swizzled_offset;整个 SF 张量已在调用前清零)
//
// 每组的计算顺序(判测按同一顺序生成真值,逐 byte 严格相等):
//   amax = 组内 16 个值的绝对值最大
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// 设备侧的 e2m1 转换直接用 cuda_fp4.h 的 __nv_fp4x2_e2m1(float2 的 .x
// 进低 nibble),它在 sm_100 家族上是单条硬件指令;你在 5.3(a) 写的
// 编码器语义与它一致,host 参考用的就是它。
//
// 组织建议:16 元素 = 32 byte,一个线程恰好负责一个组,天然免掉组内
// 线程协作;quant 没有行间依赖,grid 怎么铺完全自由。
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                   uint8_t* __restrict__ dataOut,
                                   uint8_t* __restrict__ sfOut, int M, int K) {
    for (int g = blockIdx.x * BLOCK + threadIdx.x; g < M * (K / 16);
         g += gridDim.x * BLOCK) {
        float v[16], amax = 0;
        const auto *src = in + size_t(g) * 16;
        uint4 raw[2] = {reinterpret_cast<const uint4 *>(src)[0],
                       reinterpret_cast<const uint4 *>(src)[1]};
        const auto *h = reinterpret_cast<const __nv_bfloat16 *>(raw);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            v[i] = __bfloat162float(h[i]);
            amax = fmaxf(amax, fabsf(v[i]));
        }
        __nv_fp8_e4m3 sf(amax / 6.0f);
        float inv = float(sf) != 0 ? 1.0f / float(sf) : 0;
        uint64_t out = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            __nv_fp4x2_e2m1 q(make_float2(v[2 * i] * inv, v[2 * i + 1] * inv));
            out |= uint64_t(q.__x) << (8 * i);
        }
        reinterpret_cast<uint64_t *>(dataOut)[g] = out;
        sfOut[sf_swizzled_offset(g / (K / 16), g % (K / 16), nvfp4_num_ktiles(K))] = sf.__x;
    }
}

// 判测和 5.4 会按这个签名调用;grid 大小你自己定,写在这里。
inline void launch_nvfp4_quant(const __nv_bfloat16* in, uint8_t* dataOut,
                               uint8_t* sfOut, int M, int K, int sms) {
    constexpr int B = 128;
    int grid = (M * (K / 16) + B - 1) / B;
    grid = grid < sms * 8 ? grid : sms * 8;
    nvfp4_quant_kernel<B><<<grid, B>>>(in, dataOut, sfOut, M, K);
}
