from MatPIM.Utilities import *
from math import ceil, log2


def BinaryMV(sim: Simulator, m: int, n: int):
    """
    Performs the MatPIM binary matrix-vector multiplication algorithm.
    :param sim: the simulation environment
    :param m: the number of rows in the matrix
    :param n: the number of columns in the matrix
    """

    # The number of bits per partition
    np = n // sim.kc

    # Clone x along rows
    VBroadcast(sim, 0, list(range(m)), list(range(np, 2 * np)))

    # Perform AND in parallel
    for j in range(np):
        XNOR(sim, j, np + j, np + j, list(range(m)))

    # Perform intra-partition popcount
    for i in range(m):
        for j in range(sim.kc):
            s = 0
            for k in range(np):
                s += sim.memory[i][sim.relToAbsCol(j, np + k)]
            for k in range(ceil(log2(np))):
                sim.memory[i][sim.relToAbsCol(j, np + k)] = bool(s & (1 << k))

    # Perform tree addition
    for i in range(m):
        s = 0
        for j in range(sim.kc):
            sp = 0
            for k in range(ceil(log2(np))):
                sp += int(sim.memory[i][sim.relToAbsCol(j, np + k)]) << k
            s += sp
        sim.memory[i][sim.c - 1] = (s >= (n // 2))
