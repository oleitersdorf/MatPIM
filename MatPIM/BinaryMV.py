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

    intermediates = list(range(2 * np, n))

    # Clone x along rows
    sim.perform(ParallelOperation(
        [Operation(GateType.INIT1, GateDirection.IN_ROW, [],
        sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(np, 2 * np)], []), torch.LongTensor(list(range(1, m))))]))

    log2_kr = sim.kr.bit_length() - 1
    for i in range(log2_kr):
        sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN,
            [sim.relToAbsRow(j, 0), sim.relToAbsRow(j, 0)],
            [sim.relToAbsRow(j + (sim.kr >> (i + 1)), 0)],
            torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(np, 2 * np)], [])))
            for j in range(0, sim.kr, 1 << (log2_kr - i))]))
    for k in range(1, m // sim.kr):
        sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN, [sim.relToAbsRow(p, 0), sim.relToAbsRow(p, 0)],
            [sim.relToAbsRow(p, k)], torch.LongTensor(sum([[sim.relToAbsCol(j, r) for j in range(sim.kc)] for r in range(np, 2 * np)], [])))
            for p in range(sim.kr)]))

    # Perform XNOR in parallel
    for j in range(np):
        XNOR(sim, j, np + j, list(range(m)))

    # Perform intra-partition popcount
    for k in range(0, np, 3):
        sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(partition, intermediates[0]), sim.relToAbsCol(partition, intermediates[1]),
             sim.relToAbsCol(partition, intermediates[2])]) for partition in range(sim.kc)]))

        sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, np + k)], [sim.relToAbsCol(j, intermediates[0])]) for j in range(sim.kc)]))
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, np + k), sim.relToAbsCol(j, np + k + 1), sim.relToAbsCol(j, np + k + 2)],
            [sim.relToAbsCol(j, intermediates[1])]) for j in range(sim.kc)]))
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, np + k + 2), sim.relToAbsCol(j, np + k + 1), sim.relToAbsCol(j, intermediates[0])],
            [sim.relToAbsCol(j, intermediates[2])]) for j in range(sim.kc)]))

        sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
            [sim.relToAbsCol(partition, np + k), sim.relToAbsCol(partition, np + k + 1),
             ]) for partition in range(sim.kc)]))

        sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, intermediates[1])], [sim.relToAbsCol(j, np + k + 1)]) for j in range(sim.kc)]))
        sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
            [sim.relToAbsCol(j, np + k + 1), sim.relToAbsCol(j, intermediates[0]), sim.relToAbsCol(j, intermediates[2])],
            [sim.relToAbsCol(j, np + k)]) for j in range(sim.kc)]))

    # Perform intra-partition addition tree
    a = (np // 3)
    while a > 1:

        for p in range(a // 2):

            # In each partition, add the numbers represented by first_num and second_num
            first_num = np + 2 * p * (np // a)
            second_num = np + (2 * p + 1) * (np // a)
            rep_size = ceil(log2(np // a))

            sim.perform(ParallelOperation(
                [Operation(GateType.INIT0, GateDirection.IN_ROW, [], [sim.relToAbsCol(j, intermediates[0])])
                 for j in range(sim.kc)]))
            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                [sim.relToAbsCol(j, intermediates[1]), sim.relToAbsCol(j, intermediates[2]),
                sim.relToAbsCol(j, intermediates[3]), sim.relToAbsCol(j, intermediates[4])])
                for j in range(sim.kc)]))

            for k in range(rep_size):

                # Legend
                carry_loc = intermediates[(0 if k % 2 == 0 else 1)]
                new_carry_loc = intermediates[(1 if k % 2 == 0 else 0)]
                not_carry_loc = intermediates[(2 if k % 2 == 0 else 3)]
                new_not_carry_loc = intermediates[(3 if k % 2 == 0 else 2)]
                temp_loc = intermediates[4]

                sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                    [sim.relToAbsCol(j, first_num + k), sim.relToAbsCol(j, second_num + k),
                     sim.relToAbsCol(j, carry_loc)], [sim.relToAbsCol(j, new_not_carry_loc)])
                    for j in range(sim.kc)]))
                sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
                    [sim.relToAbsCol(j, new_not_carry_loc)], [sim.relToAbsCol(j, new_carry_loc)])
                    for j in range(sim.kc)]))
                sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                    [sim.relToAbsCol(j, first_num + k), sim.relToAbsCol(j, second_num + k),
                     sim.relToAbsCol(j, not_carry_loc)], [sim.relToAbsCol(j, temp_loc)])
                    for j in range(sim.kc)]))

                sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                    [sim.relToAbsCol(j, first_num + k)] + ([sim.relToAbsCol(j, first_num + k + 1)] if k == (rep_size - 1) else []))
                    for j in range(sim.kc)]))

                sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                    [sim.relToAbsCol(j, new_carry_loc), sim.relToAbsCol(j, not_carry_loc),
                    sim.relToAbsCol(j, temp_loc)], [sim.relToAbsCol(j, first_num + k)])
                    for j in range(sim.kc)]))

                if k == (rep_size - 1):
                    sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
                        [sim.relToAbsCol(j, new_not_carry_loc)], [sim.relToAbsCol(j, first_num + k + 1)])
                        for j in range(sim.kc)]))

                sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                    [sim.relToAbsCol(j, temp_loc), sim.relToAbsCol(j, carry_loc), sim.relToAbsCol(j, not_carry_loc)])
                    for j in range(sim.kc)]))
        a //= 2

    # Perform tree addition
    a = sim.kc
    while a > 1:

        first_nums = [2*p*(sim.kc // a) for p in range(a // 2)]
        second_nums = [(2*p+1)*(sim.kc // a) for p in range(a // 2)]
        rep_size = ceil(log2(np)) + (ceil(log2(sim.kc // a)))

        sim.perform(ParallelOperation(
                [Operation(GateType.INIT0, GateDirection.IN_ROW, [], [sim.relToAbsCol(first_nums[p], intermediates[0])])
                 for p in range(a // 2)]))
        sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                [sim.relToAbsCol(first_nums[p], intermediates[1]), sim.relToAbsCol(first_nums[p], intermediates[2]),
                sim.relToAbsCol(first_nums[p], intermediates[3]), sim.relToAbsCol(first_nums[p], intermediates[4])])
                for p in range(a // 2)]))

        for k in range(rep_size):

            # Legend
            carry_loc = intermediates[(0 if k % 2 == 0 else 1)]
            new_carry_loc = intermediates[(1 if k % 2 == 0 else 0)]
            not_carry_loc = intermediates[(2 if k % 2 == 0 else 3)]
            new_not_carry_loc = intermediates[(3 if k % 2 == 0 else 2)]
            temp_loc = intermediates[4]

            sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                [sim.relToAbsCol(first_nums[p], np + k), sim.relToAbsCol(second_nums[p], np + k),
                 sim.relToAbsCol(first_nums[p], carry_loc)], [sim.relToAbsCol(first_nums[p], new_not_carry_loc)])
                for p in range(a // 2)]))
            sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
                [sim.relToAbsCol(first_nums[p], new_not_carry_loc)], [sim.relToAbsCol(first_nums[p], new_carry_loc)])
                for p in range(a // 2)]))
            sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                [sim.relToAbsCol(first_nums[p], np + k), sim.relToAbsCol(second_nums[p], np + k),
                 sim.relToAbsCol(first_nums[p], not_carry_loc)], [sim.relToAbsCol(first_nums[p], temp_loc)])
                for p in range(a // 2)]))

            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                [sim.relToAbsCol(first_nums[p], np + k)] + ([sim.relToAbsCol(first_nums[p], np + k + 1)] if k == (rep_size - 1) else []))
                for p in range(a // 2)]))

            sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                [sim.relToAbsCol(first_nums[p], new_carry_loc), sim.relToAbsCol(first_nums[p], not_carry_loc),
                sim.relToAbsCol(first_nums[p], temp_loc)], [sim.relToAbsCol(first_nums[p], np + k)])
                for p in range(a // 2)]))

            if k == (rep_size - 1):
                sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
                    [sim.relToAbsCol(first_nums[p], new_not_carry_loc)], [sim.relToAbsCol(first_nums[p], np + k + 1)])
                    for p in range(a // 2)]))

            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                [sim.relToAbsCol(first_nums[p], temp_loc), sim.relToAbsCol(first_nums[p], carry_loc), sim.relToAbsCol(first_nums[p], not_carry_loc)])
                for p in range(a // 2)]))

        a //= 2
