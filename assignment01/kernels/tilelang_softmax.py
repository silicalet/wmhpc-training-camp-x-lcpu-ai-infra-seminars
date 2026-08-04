"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import functools

import torch
import tilelang
import tilelang.language as T


def make_softmax(M: int, N: int, dtype="float32"):
    block_n = 1 << (N - 1).bit_length()
    threads = 256 if block_n >= 256 else 128

    @T.prim_func
    def softmax_kernel(
        x: T.Buffer((M, N), dtype),
        y: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(M, threads=threads) as bx:
            x_local = T.alloc_fragment((1, block_n), dtype)
            rmax = T.alloc_fragment((1,), dtype)
            rsum = T.alloc_fragment((1,), dtype)

            for _, j in T.Parallel(1, block_n):
                x_local[0, j] = T.if_then_else(j < N, x[bx, j], -T.infinity(dtype))

            T.reduce_max(x_local, rmax, dim=1)

            for _, j in T.Parallel(1, block_n):
                x_local[0, j] = T.exp(x_local[0, j] - rmax[0])

            T.reduce_sum(x_local, rsum, dim=1)

            for _, j in T.Parallel(1, N):
                y[bx, j] = x_local[0, j] / rsum[0]

    return softmax_kernel


@functools.lru_cache(maxsize=None)
def _compiled_softmax(M: int, N: int):
    return tilelang.compile(make_softmax(M, N), out_idx=[1])


def softmax(x: torch.Tensor) -> torch.Tensor:
    return _compiled_softmax(x.shape[0], x.shape[1])(x)
