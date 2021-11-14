import torch.nn.functional
from MatPIM.Utilities import *
from tqdm import tqdm


def FullPrecisionConv(sim: Simulator, m: int, n: int, k: int, alpha: int):
    """
    Performs the MatPIM full-precision convolution algorithm.
    :param sim: the simulation environment
    :param m: the number of rows in the matrix
    :param n: the number of columns in the matrix
    :param k: the side length of the convolution matrix
    :param alpha: the number of blocks
    """

    # K initially stored (packed) in reg 0, unpacked is stored in reg 1
    k_packed = 0
    k_unpacked = 1
    # Overlap stored in regs 2 to 1 + k
    overlap = 2
    # A (without overlap) stored in regs 1 + k to 1 + k + n // alpha
    A_start = 1 + k
    # AK stored in regs 1 + k + n // alpha to 1 + k + 2 * n // alpha
    AK_start = 1 + k + n // alpha

    if alpha > 1:
        # Copy and shift columns
        for a in range(k-1):
            Move(sim, A_start + (n // alpha - (k - 1)) + a, overlap + a)
        # Shift vertically
        for i in reversed(range(m, m*alpha)):
            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(overlap, overlap + (k - 1))], []),
                torch.LongTensor([i]))]))
            sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN, [i - m, i - m], [i],
                torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(overlap, overlap + (k - 1))], [])))]))
        sim.perform(ParallelOperation([Operation(GateType.INIT0, GateDirection.IN_ROW, [],
                sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(overlap, overlap + (k - 1))], []),
                torch.LongTensor(list(range(m))))]))

        for vert in range(k):
            for hori in tqdm(range(k)):

                Move(sim, k_packed, k_unpacked, torch.LongTensor([vert * k + hori]))
                VBroadcast(sim, vert * k + hori, [x for x in list(range(m * alpha)) if x != (vert * k + hori)], [k_unpacked])

                for col in range(((k-1) + n // alpha) - (k - 1)):
                    Multiply(sim, overlap + col + hori, k_unpacked, AK_start + col, AK_start + col)

            # Shift vertically
            for a in range(alpha):
                for i in range(a*m, a*m + m-1):
                    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                        sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(overlap, overlap + (k-1) + n // alpha)], []),
                        torch.LongTensor([i]))]))
                    sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN, [i + 1, i + 1], [i],
                        torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(overlap, overlap + (k-1) + n // alpha)], [])))]))

    else:

        for vert in range(k):
            for hori in tqdm(range(k)):

                Move(sim, k_packed, k_unpacked, torch.LongTensor([vert * k + hori]))
                VBroadcast(sim, vert * k + hori, [x for x in list(range(m * alpha)) if x != (vert * k + hori)], [k_unpacked])

                for col in range((n // alpha) - (k - 1)):
                    Multiply(sim, A_start + col + hori, k_unpacked, AK_start + col, AK_start + col)

            # Shift vertically
            for i in range(m-1):
                sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                    sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(overlap, overlap + (k-1) + n // alpha)], []),
                    torch.LongTensor([i]))]))
                sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN, [i + 1, i + 1], [i],
                    torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(overlap, overlap + (k-1) + n // alpha)], [])))]))