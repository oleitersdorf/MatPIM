import torch
from typing import List
from Simulator.simulator import Simulator, ParallelOperation, Operation, GateType, GateDirection


def MoveNOT(sim: Simulator, a: int, b: int, mask=None):
    """
    Copies the value from register a to register b in all rows of the mask, copies notted
    :param sim: the simulation environment
    :param a: the intra-partition index of the input
    :param b: the intra-partition index of the output
    :param mask: the row mask
    """

    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        [sim.relToAbsCol(j, b) for j in range(sim.kc)], mask)]))
    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, a)], [sim.relToAbsCol(j, b)], mask) for j in range(sim.kc)]))


def VCOPY(sim: Simulator, a_s: List[int], b_s: List[int], regs: List[int]):
    """
    Copies from rows a_s to rows b_s the values of registers regs, copies notted
    :param sim: the simulation environment
    :param a_s: the list of input rows
    :param b_s: the list of output rows
    :param regs: the registers to move
    """

    assert (len(a_s) == len(b_s))

    sim.perform(ParallelOperation(
        [Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in regs], []), torch.LongTensor(b_s))]))

    for a, b in zip(a_s, b_s):
        sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_COLUMN, [a], [b],
        torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in regs], [])))]))


def VBroadcast(sim: Simulator, a: int, b_s: List[int], regs: List[int]):
    """
    Broadcasts from row a to set of rows b_s the values of registers regs
    :param sim: the simulation environment
    :param a: the row to broadcast from
    :param b_s: the list of output rows
    :param regs: the registers to move
    """

    assert(len(b_s) >= 2)

    sim.perform(ParallelOperation(
        [Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                   sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in regs], []), torch.LongTensor(b_s + [sim.r]))]))

    # Copy first to intermediate row (sim.r), then from sim.r to rest (as NOT and not copy)

    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_COLUMN, [a], [sim.r],
        torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in regs], [])))]))

    for b in b_s:
        sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_COLUMN, [sim.r], [b],
        torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in regs], [])))]))


def XNOR(sim: Simulator, a: int, b: int, mask=None, intermediates=None):
    """
    Performs a row-parallel XNOR on numbers stored in indices a and b, storing the result in b.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number, and the output
    :param mask: the row mask
    :param intermediates: intermediate registers than are used
    """

    if intermediates is None:
        intermediates = [sim.num_regs - 3, sim.num_regs - 2, sim.num_regs - 1]

    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        sum([[sim.relToAbsCol(j, intermediates[0]), sim.relToAbsCol(j, intermediates[1]),
              sim.relToAbsCol(j, intermediates[2])] for j in range(sim.kc)], []), mask)]))

    sim.perform(ParallelOperation([Operation(GateType.NOR, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, a), sim.relToAbsCol(j, b)], [sim.relToAbsCol(j, intermediates[0])], mask) for j in range(sim.kc)]))
    sim.perform(ParallelOperation([Operation(GateType.NOR, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, a), sim.relToAbsCol(j, intermediates[0])], [sim.relToAbsCol(j, intermediates[1])], mask) for j in range(sim.kc)]))
    sim.perform(ParallelOperation([Operation(GateType.NOR, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, b), sim.relToAbsCol(j, intermediates[0])], [sim.relToAbsCol(j, intermediates[2])], mask) for j in range(sim.kc)]))
    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        [sim.relToAbsCol(j, b) for j in range(sim.kc)], mask)]))
    sim.perform(ParallelOperation([Operation(GateType.NOR, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, intermediates[1]), sim.relToAbsCol(j, intermediates[2])], [sim.relToAbsCol(j, b)], mask) for j in range(sim.kc)]))


def Add(sim: Simulator, a: int, b: int, z: int, mask=None):
    """
    Performs a row-parallel addition on numbers stored in indices a and b, storing the result in z.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number
    :param z: the intra-partition index of the output
    :param mask: the row mask
    """

    for i in mask:
        a_val = sim.loadIntegerStrided(a, i)
        b_val = sim.loadIntegerStrided(b, i)

        sim.storeIntegerStrided(a_val + b_val, z, i)

    # TODO
    sim.latency += 160
    sim.energy += 160


def Multiply(sim: Simulator, a: int, b: int, z: int, c=None, mask=None):
    """
    Performs a row-parallel multiplication on numbers stored in indices a and b, storing the result in z. If c is not None,
    then computes instead z = c + (a * b) with identical latency.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number
    :param z: the intra-partition index of the output
    :param c: the intra-partition index of an additional sum
    :param mask: the row mask
    """

    for i in mask:
        a_val = sim.loadIntegerStrided(a, i)
        b_val = sim.loadIntegerStrided(b, i)
        c_val = sim.loadIntegerStrided(c, i) if c else 0

        sim.storeIntegerStrided(a_val * b_val + c_val, z, i)

    # TODO
    sim.latency += 400
    sim.energy += 7500


def InnerProduct(sim: Simulator, n: int, x: List[int], y: List[int], z: int, mask=None):
    """
    Computes the inner product in the given rows between vectors x and y (given as intra-partition indices), storing in z
    :param sim: the simulation environment
    :param n: the dimension of the inner product
    :param x: the intra-partition indices of the first vector
    :param y: the intra-partition indices of the second vector
    :param z: the intra-partition index of the output vector
    :param mask: the row mask
    """

    for i in range(n):
        Multiply(sim, x[i], y[i], z, z, mask)
