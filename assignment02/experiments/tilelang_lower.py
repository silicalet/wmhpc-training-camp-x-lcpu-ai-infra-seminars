"""6.1: compile one T.gemm for both targets and retain every lowering artifact."""
import pathlib
import subprocess

import tilelang
import tilelang.language as T
from tilelang import tvm


@T.prim_func
def gemm(a: T.Tensor((1024, 1024), 'float16'),
         b: T.Tensor((1024, 1024), 'float16'),
         c: T.Tensor((1024, 1024), 'float32')):
    with T.Kernel(8, 8, threads=128) as (i, j):
        sa = T.alloc_shared((128, 64), 'float16')
        sb = T.alloc_shared((64, 128), 'float16')
        acc = T.alloc_fragment((128, 128), 'float32')
        T.clear(acc)
        for k in T.Pipelined(16, num_stages=3):
            T.copy(a[i * 128, k * 64], sa)
            T.copy(b[k * 64, j * 128], sb)
            T.gemm(sa, sb, acc)
        T.copy(acc, c[i * 128, j * 128])


def main():
    out = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'tilelang'
    out.mkdir(parents=True, exist_ok=True)
    print('TileLang', tilelang.__version__)
    (out / 'input.tir').write_text(gemm.script())
    for arch in ['sm_90a', 'sm_100a']:
        target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
        with target:
            mod = tilelang.lower(gemm, target=target)
        (out / f'{arch}.cu').write_text(mod.kernel_source)
        (out / f'{arch}.tir').write_text(mod.device_mod.script())
        (out / f'{arch}.host.tir').write_text(mod.host_mod.script())
        pkg = pathlib.Path(tilelang.__file__).parent
        cmd = ['nvcc', '-O2', '-std=c++17', '--expt-relaxed-constexpr',
               f'-I{pkg / "src"}', f'-I{pkg / "3rdparty/cutlass/include"}',
               '-gencode', f'arch=compute_{arch[3:]},code={arch}',
               '-cubin', str(out / f'{arch}.cu'), '-o', str(out / f'{arch}.cubin')]
        run = subprocess.run(cmd, text=True, capture_output=True)
        (out / f'{arch}-build.log').write_text(' '.join(cmd) + '\n' + run.stdout + run.stderr + f'\nexit={run.returncode}\n')
        run.check_returncode()
        print(arch, 'lowered', type(mod).__name__)


if __name__ == '__main__':
    main()
