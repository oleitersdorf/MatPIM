import torch
from typing import List
from Simulator.simulator import Simulator, ParallelOperation, Operation, GateType, GateDirection


def Move(sim: Simulator, a: int, b: int, mask=None):
    """
    Copies the value from register a to register b in all rows of the mask
    :param sim: the simulation environment
    :param a: the intra-partition index of the input
    :param b: the intra-partition index of the output
    :param mask: the row mask
    """

    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        [sim.relToAbsCol(j, b) for j in range(sim.kc)], mask)]))
    sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, a), sim.relToAbsCol(j, a)], [sim.relToAbsCol(j, b)], mask) for j in range(sim.kc)]))


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


def AND(sim: Simulator, a: int, b: int, mask=None, intermediates=None):
    """
    Performs a row-parallel AND on numbers stored in indices a and b, storing the result in b.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number, and the output
    :param mask: the row mask
    :param intermediates: intermediate registers that are used
    """

    if intermediates is None:
        intermediates = [sim.num_regs - 1]

    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        sum([[sim.relToAbsCol(j, intermediates[0])] for j in range(sim.kc)], []), mask)]))

    sim.perform(ParallelOperation([Operation(GateType.NAND, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, a), sim.relToAbsCol(j, b)], [sim.relToAbsCol(j, intermediates[0])], mask) for j in range(sim.kc)]))
    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        [sim.relToAbsCol(j, b) for j in range(sim.kc)], mask)]))
    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, intermediates[0])], [sim.relToAbsCol(j, b)], mask) for j in range(sim.kc)]))


def OR(sim: Simulator, a: int, b: int, mask=None, intermediates=None):
    """
    Performs a row-parallel OR on numbers stored in indices a and b, storing the result in b.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number, and the output
    :param mask: the row mask
    :param intermediates: intermediate registers that are used
    """

    if intermediates is None:
        intermediates = [sim.num_regs - 1]

    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        sum([[sim.relToAbsCol(j, intermediates[0])] for j in range(sim.kc)], []), mask)]))

    sim.perform(ParallelOperation([Operation(GateType.NOR, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, a), sim.relToAbsCol(j, b)], [sim.relToAbsCol(j, intermediates[0])], mask) for j in range(sim.kc)]))
    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        [sim.relToAbsCol(j, b) for j in range(sim.kc)], mask)]))
    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
        [sim.relToAbsCol(j, intermediates[0])], [sim.relToAbsCol(j, b)], mask) for j in range(sim.kc)]))


def XNOR(sim: Simulator, a: int, b: int, mask=None, intermediates=None):
    """
    Performs a row-parallel XNOR on numbers stored in indices a and b, storing the result in b.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number, and the output
    :param mask: the row mask
    :param intermediates: intermediate registers that are used
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


def Add(sim: Simulator, a: int, b: int, mask=None, intermediate=None):
    """
    Performs a bit-serial row-parallel addition on numbers stored in indices a and b, storing the result in b.
    Based on the MultPIM algorithm for serial addition.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number, and the output register
    :param mask: the row mask
    :param intermediate: intermediate register that used
    """

    if intermediate is None:
        intermediate = sim.num_regs - 1

    sim.perform(ParallelOperation([Operation(GateType.INIT0, GateDirection.IN_ROW, [], [sim.relToAbsCol(0, intermediate)], mask)]))
    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        [sim.relToAbsCol(1, intermediate), sim.relToAbsCol(2, intermediate), sim.relToAbsCol(3, intermediate), sim.relToAbsCol(4, intermediate)], mask)]))

    for k in reversed(range(sim.kc)):

        # Legend
        carry_loc = (0 if k % 2 == 1 else 1)
        new_carry_loc = (1 if k % 2 == 1 else 0)
        not_carry_loc = (2 if k % 2 == 1 else 3)
        new_not_carry_loc = (3 if k % 2 == 1 else 2)
        temp_loc = 4

        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(k, a), sim.relToAbsCol(k, b), sim.relToAbsCol(carry_loc, intermediate)], [sim.relToAbsCol(new_not_carry_loc, intermediate)], mask)]))
        sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(new_not_carry_loc, intermediate)], [sim.relToAbsCol(new_carry_loc, intermediate)], mask)]))
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(k, a), sim.relToAbsCol(k, b), sim.relToAbsCol(not_carry_loc, intermediate)], [sim.relToAbsCol(temp_loc, intermediate)], mask)]))

        sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(k, b)], mask)]))

        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(new_carry_loc, intermediate), sim.relToAbsCol(not_carry_loc, intermediate),
            sim.relToAbsCol(temp_loc, intermediate)], [sim.relToAbsCol(k, b)], mask)]))

        sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(temp_loc, intermediate), sim.relToAbsCol(carry_loc, intermediate), sim.relToAbsCol(not_carry_loc, intermediate)], mask)]))


def Multiply(sim: Simulator, a: int, b: int, z: int, c=None, mask=None, intermediates=None):
    """
    Performs a row-parallel multiplication on numbers stored in indices a and b, storing the result in z. If c is not None,
    then computes instead z = c + (a * b) with identical latency.
    Based on the MultPIM algorithm for parallel multiplication.
    :param sim: the simulation environment
    :param a: the intra-partition index of the first number
    :param b: the intra-partition index of the second number
    :param z: the intra-partition index of the output
    :param c: the intra-partition index of an additional sum
    :param mask: the row mask
    :param intermediates: the intermediates used
    """

    if intermediates is None:
        intermediates = list(range(sim.num_regs-11 if c is None else sim.num_regs-10, sim.num_regs))

    # Legend
    ABIT = intermediates[0]
    BBIT = intermediates[1]
    ABBIT = intermediates[2]
    TEMP = intermediates[3]
    CBITEven = intermediates[4]
    NotCBITEven = intermediates[5]
    SBITOdd = intermediates[6]
    CBITOdd = intermediates[7]
    NotCBITOdd = intermediates[8]
    OUTPUT = intermediates[9]
    SBITEven = intermediates[10] if c is None else c

    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        [sim.relToAbsCol(partition, BBIT), sim.relToAbsCol(partition, ABBIT), sim.relToAbsCol(partition, NotCBITEven),
         sim.relToAbsCol(partition, SBITOdd), sim.relToAbsCol(partition, CBITOdd), sim.relToAbsCol(partition, NotCBITOdd),
         sim.relToAbsCol(partition, TEMP), sim.relToAbsCol(partition, OUTPUT), sim.relToAbsCol(partition, ABIT)
         ], mask)
        for partition in range(sim.kc)]))

    if c is None:
        sim.perform(ParallelOperation([Operation(GateType.INIT0, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(partition, SBITEven), sim.relToAbsCol(partition, CBITEven),
             ], mask)
            for partition in range(sim.kc)]))
    else:
        sim.perform(ParallelOperation([Operation(GateType.INIT0, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(partition, CBITEven),
             ], mask)
            for partition in range(sim.kc)]))

    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, a)], [sim.relToAbsCol(j, ABIT)], mask) for j in range(sim.kc)]))

    # Iterate over all N stages
    for k in range(sim.kc):

        # The bit locations relevant to this iteration
        iterSBIT = SBITEven if k % 2 == 0 else SBITOdd
        iterCBIT = CBITEven if k % 2 == 0 else CBITOdd
        iterNotCBIT = NotCBITEven if k % 2 == 0 else NotCBITOdd
        nextSBIT = SBITEven if k % 2 == 1 else SBITOdd
        nextCBIT = CBITEven if k % 2 == 1 else CBITOdd
        nextNotCBIT = NotCBITEven if k % 2 == 1 else NotCBITOdd

        # Copy b_k to all partitions using log_2(N) ops
        # --- log_2(N) OPs --- #
        sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(sim.kc - k - 1, b)], [sim.relToAbsCol(0, BBIT)], mask)]))
        log2_N = sim.kc.bit_length() - 1
        for i in range(log2_N):
            sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_ROW,
                [sim.relToAbsCol(j, BBIT), sim.relToAbsCol(j, BBIT)], [sim.relToAbsCol(j + (sim.kc >> (i+1)), BBIT)], mask)
                    for j in range(0, sim.kc, 1 << (log2_N - i))]))

        # Compute partial products
        # --- 1 OP --- #
        sim.perform(ParallelOperation([Operation(GateType.NOR, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, ABIT), sim.relToAbsCol(j, BBIT)],
            [sim.relToAbsCol(j, ABBIT)], mask) for j in range(sim.kc)]))

        # Compute new not(carry)
        # --- 1 OP --- #
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, ABBIT), sim.relToAbsCol(j, iterSBIT),
            sim.relToAbsCol(j, iterCBIT)], [sim.relToAbsCol(j, nextNotCBIT)], mask) for j in range(sim.kc)]))
        # Compute new carry
        # --- 1 OP --- #
        sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, nextNotCBIT)], [sim.relToAbsCol(j, nextCBIT)], mask) for j in range(sim.kc)]))

        # Compute Min3(AB, S, not(C))
        # --- 1 OP --- #
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, ABBIT), sim.relToAbsCol(j, iterSBIT),
            sim.relToAbsCol(j, iterNotCBIT)], [sim.relToAbsCol(j, TEMP)], mask) for j in range(sim.kc)]))

        # Compute S across adjacent partitions
        # --- 2 OPs --- #
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, nextCBIT), sim.relToAbsCol(j, iterNotCBIT),
            sim.relToAbsCol(j, TEMP)], [sim.relToAbsCol(j + 1, nextSBIT)], mask) for j in range(0, sim.kc - 1, 2)]))
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, nextCBIT), sim.relToAbsCol(j, iterNotCBIT),
            sim.relToAbsCol(j, TEMP)], [sim.relToAbsCol(j + 1, nextSBIT)], mask) for j in range(1, sim.kc - 1, 2)]))
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(sim.kc - 1, nextCBIT), sim.relToAbsCol(sim.kc - 1, iterNotCBIT),
            sim.relToAbsCol(sim.kc - 1, TEMP)], [sim.relToAbsCol(sim.kc - k - 1, OUTPUT)], mask)]))

        # Init the temps for next time
        # --- 1 OP --- #
        sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(partition, BBIT), sim.relToAbsCol(partition, ABBIT), sim.relToAbsCol(partition, iterSBIT),
             sim.relToAbsCol(partition, iterCBIT), sim.relToAbsCol(partition, iterNotCBIT), sim.relToAbsCol(partition, TEMP)], mask)
            for partition in range(sim.kc)]))

    sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(partition, TEMP), sim.relToAbsCol(partition, z)], mask)
            for partition in range(sim.kc)]))

    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, OUTPUT)], [sim.relToAbsCol(j, TEMP)], mask) for j in range(sim.kc)]))
    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, TEMP)], [sim.relToAbsCol(j, z)], mask) for j in range(sim.kc)]))


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

    Multiply(sim, x[0], y[0], z, None, mask)
    for i in range(1, n):
        Multiply(sim, x[i], y[i], z, z, mask)
