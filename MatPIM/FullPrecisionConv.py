import torch.nn.functional
from MatPIM.Utilities import *


def FullPrecisionConv(sim: Simulator, m: int, n: int, k: int):
    """
    Performs the MatPIM full-precision convolution algorithm.
    :param sim: the simulation environment
    :param m: the number of rows in the matrix
    :param n: the number of columns in the matrix
    :param k: the side length of the convolution matrix
    """

    # K initially stored (packed) in reg 0, unpacked is stored in reg 1
    k_packed = 0
    k_unpacked = 1
    # A initially stored in regs 3 to 3 + n
    A_start_center = 3
    # AK stored in regs 3 + n + 1 to 3 + n + 1 + n - (k // 2 + 1)
    AK_start = 3 + n + 1

    for vert in range(k):
        for hori in range(k):

            Move(sim, k_packed, k_unpacked, torch.LongTensor([vert * k + hori]))
            VBroadcast(sim, vert * k + hori, [x for x in list(range(m)) if x != (vert * k + hori)], [k_unpacked])

            for col in range(n - (k // 2 + 1)):
                Multiply(sim, A_start_center + col + hori, k_unpacked, AK_start + col, AK_start + col)

        # Shift vertically
        for i in range(m-1):
            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(A_start_center, A_start_center + n)], []),
                torch.LongTensor([i]))]))
            sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN, [i + 1, i + 1], [i],
                torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(A_start_center, A_start_center + n)], [])))]))
