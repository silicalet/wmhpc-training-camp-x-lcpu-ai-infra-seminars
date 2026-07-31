"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""

lane: list[int] = [0] * 32


def xcpc(program, ln):
    assert type(program) == list
    cycle = 0

    for op in program:
        assert type(op) == tuple
        if op[0] == "add":
            cycle += 1
            k: int = op[1]
            for i in ln:
                lane[i] += k
        elif op[0] == "mul":
            cycle += 1
            k: int = op[1]
            for i in ln:
                lane[i] *= k
        else:
            t, then_prog, else_prog = op[1:]
            a: list[int] = []
            b: list[int] = []
            for i in ln:
                if lane[i] < t:
                    a.append(i)
                else:
                    b.append(i)

            if a:
                cycle += xcpc(then_prog, a)[1]
            if b:
                cycle += xcpc(else_prog, b)[1]

    return (lane, cycle)


def run(program):
    lane[:] = list(range(32))
    return xcpc(program, list(range(32)))
