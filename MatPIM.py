import torch
from typing import List
from simulator import Simulator, ParallelOperation, Operation, GateType, GateDirection


def Move(sim: Simulator, a: int, b: int, mask=None):
    """
    Copies the value from register a to register b in all rows of the mask
    :param sim: the simulation environment
    :param a: the intra-partition index of the input
    :param b: the intra-partition index of the output
    :param mask: the row mask
    """

    for i in mask:

        sim.storeIntegerStrided(sim.loadIntegerStrided(a, i), b, i)

    # TODO
    sim.latency += 2
    sim.energy += 64


def VCOPY(sim: Simulator, a_s: List[int], b_s: List[int], regs: List[int]):
    """
    Copies from rows a_s to rows b_s the values of registers regs
    :param sim: the simulation environment
    :param a_s: the list of input rows
    :param b_s: the list of output rows
    :param regs: the registers to move
    """

    assert(len(a_s) == len(b_s))

    for a, b in zip(a_s, b_s):
        for reg in regs:
            sim.storeIntegerStrided(sim.loadIntegerStrided(reg, a), reg, b)

    # TODO
    sim.latency += 1 + len(a_s)
    sim.energy += len(a_s) * len(regs) * 32 * 2


def VBroadcast(sim: Simulator, a: int, b_s: List[int], regs: List[int]):
    """
    Broadcasts from row a to set of rows b_s the values of registers regs
    :param sim: the simulation environment
    :param a: the row to broadcast from
    :param b_s: the list of output rows
    :param regs: the registers to move
    """
    for b in b_s:
        for reg in regs:
            sim.storeIntegerStrided(sim.loadIntegerStrided(reg, a), reg, b)

    # TODO
    sim.latency += 1 + len(b_s)
    sim.energy += len(b_s) * len(regs) * 32


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


def FullPrecisionMV(sim: Simulator, m: int, n: int, alpha: int):
    """
    Performs the MatPIM full-precision matrix-vector multiplication algorithm.
    :param sim: the simulation environment
    :param m: the number of rows in the matrix
    :param n: the number of columns in the matrix
    :param alpha: the number of blocks
    """

    # The effective number of elements per row from the vector
    na = n // alpha

    # Clone x along rows
    for a in range(alpha):
        VBroadcast(sim, a * m, list(range(a * m, a * m + m)), list(range(na, 2*na)))

    # Perform inner product in parallel
    InnerProduct(sim, na, list(range(na)), list(range(na, 2*na)), 2*na, list(range(alpha * m)))

    # Perform reduction
    a = alpha
    while a > 1:

        blocks_to_shift = range(alpha//a, alpha, 2*alpha//a)
        blocks_to_receive = range(0, alpha, 2*alpha//a)

        # Shift right
        Move(sim, 2*na, 2*na + 1, mask=list(range(alpha * m)))

        # Perform VCOPY
        VCOPY(sim, sum([list(range(start*m, start*m+m)) for start in blocks_to_shift], []), sum([list(range(start*m, start*m+m)) for start in blocks_to_receive], []), [2*na + 1])

        # Perform addition
        Add(sim, 2*na, 2*na + 1, 2*na, mask=list(range(alpha * m)))

        a //= 2
