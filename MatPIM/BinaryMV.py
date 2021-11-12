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
    VBroadcast(sim, 0, list(range(1, m)), list(range(np, 2 * np)))
    sim.latency = 40  # TODO (partition-based broadcast)

    # Perform XNOR in parallel
    for j in range(np):
        XNOR(sim, j, np + j, list(range(m)))

    # Perform intra-partition popcount
    sim.latency += 6*(np // 3)  # TODO
    sim.latency += 6 * 7
    for i in range(m):
        for j in range(sim.kc):

            # Divide into groups of three
            for k in range(0, np, 3):
                a = sim.memory[i][sim.relToAbsCol(j, np + k)]
                b = sim.memory[i][sim.relToAbsCol(j, np + k + 1)]
                c = sim.memory[i][sim.relToAbsCol(j, np + k + 2)]
                sout = a ^ b ^ c
                cout = (a & b) | (a & c) | (b & c)
                sim.memory[i][sim.relToAbsCol(j, np + k)] = sout
                sim.memory[i][sim.relToAbsCol(j, np + k + 1)] = cout
                # sim.latency += 6

            # Perform intra-partition addition tree
            # a = (np // 3)
            # while a > 1:
            #
            #     for k in range(a // 2):
            #
            #         # Add the numbers in [np + k*(np//a) : np + k*(np//a) + ceil(log2(np // a))] and
            #         # [np + k*(np//a) : np + k*(np//a) + ceil(log2(np // a))]
            #
            #     a /= 2

            s = 0
            for k in range(0, np, 3):
                s += (sim.memory[i][sim.relToAbsCol(j, np + k)]) + (int(sim.memory[i][sim.relToAbsCol(j, np + k + 1)]) << 1)
            # sim.latency += 6 * 7
            for k in range(ceil(log2(np))):
                sim.memory[i][sim.relToAbsCol(j, np + k)] = bool(s & (1 << k))

    # Perform tree addition
    sim.latency += 30 * 5
    for i in range(m):
        s = 0
        for j in range(sim.kc):
            sp = 0
            for k in range(ceil(log2(np))):
                sp += int(sim.memory[i][sim.relToAbsCol(j, np + k)]) << k
            s += sp
        sim.memory[i][sim.c - 1] = (s >= (n // 2))
