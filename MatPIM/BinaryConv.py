from MatPIM.Utilities import *
from math import ceil, log2


def BinaryConv(sim: Simulator, m: int, n: int, k: int):
    """
    Performs the MatPIM binary convolution algorithm.
    :param sim: the simulation environment
    :param m: the number of rows in the input matrix
    :param n: the number of columns in the input matrix
    :param k: the side length of the kernel
    """

    na = n // sim.kc

    # Legend
    overlap = 0
    A_start = overlap + k - 1
    AK_start = A_start + na
    running_start = AK_start + na
    K_locs = running_start + ceil(log2(k ** 2))
    K_unpacked_spread_start = K_locs + k
    intermediates = [K_unpacked_spread_start + 1, K_unpacked_spread_start + 2, K_unpacked_spread_start + 3,
                     K_unpacked_spread_start + 4, K_unpacked_spread_start + 5]

    # Clone overlap columns
    sim.perform(ParallelOperation(
        [Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                   sum([[sim.relToAbsCol(i, r) for r in range(overlap, overlap + k - 1)] for i in range(sim.kc)], []))]))
    sim.perform(ParallelOperation(
        [Operation(GateType.INIT0, GateDirection.IN_ROW, [],
                   sum([[sim.relToAbsCol(i, r) for r in range(overlap, overlap + k - 1)] for i in [0]], []))]))
    for a in range(k-1):
        sim.perform(ParallelOperation(
            [Operation(GateType.OR, GateDirection.IN_ROW, [sim.relToAbsCol(i, A_start + na - (k - 1) + a),
                sim.relToAbsCol(i, A_start + na - (k - 1) + a)], [sim.relToAbsCol(i + 1, overlap + a)])
            for i in range(0, sim.kc - 1, 2)]))
        sim.perform(ParallelOperation(
            [Operation(GateType.OR, GateDirection.IN_ROW, [sim.relToAbsCol(i, A_start + na - (k - 1) + a),
                sim.relToAbsCol(i, A_start + na - (k - 1) + a)], [sim.relToAbsCol(i + 1, overlap + a)])
             for i in range(1, sim.kc - 1, 2)]))

    # Unpack K from the last column to k**2 columns throughout
    sim.perform(ParallelOperation(
        [Operation(GateType.INIT1, GateDirection.IN_ROW, [], [sim.relToAbsCol(i, K_unpacked_spread_start) for i in range(k ** 2)])]))
    for j in range(k**2):
        sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_ROW,
            [sim.c, sim.c], [sim.relToAbsCol(j, K_unpacked_spread_start)], torch.LongTensor([j]))]))
        sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN,
            [j, j], [0], torch.LongTensor([sim.relToAbsCol(j, K_unpacked_spread_start)]))]))
    log2_kr = sim.kr.bit_length() - 1
    for ll in range(log2_kr):
        sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN,
            [sim.relToAbsRow(j, 0), sim.relToAbsRow(j, 0)],
            [sim.relToAbsRow(j + (sim.kr >> (ll + 1)), 0)],
            torch.LongTensor([sim.relToAbsCol(j, K_unpacked_spread_start) for j in range(k ** 2)]))
            for j in range(0, sim.kr, 1 << (log2_kr - ll))]))
    for ka in range(1, m // sim.kr):
        sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN, [sim.relToAbsRow(p, 0), sim.relToAbsRow(p, 0)],
            [sim.relToAbsRow(p, ka)], torch.LongTensor([sim.relToAbsCol(j, K_unpacked_spread_start) for j in range(k ** 2)]))
            for p in range(sim.kr)]))

    for a in range(na):

        # Initialize the running sum to zero
        sim.perform(ParallelOperation(
            [Operation(GateType.INIT0, GateDirection.IN_ROW, [],
            sum([[sim.relToAbsCol(i, r) for r in range(running_start, running_start + ceil(log2(k ** 2)))]
            for i in range(sim.kc)], []))]))

        for vert in range(k):

            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                sum([[sim.relToAbsCol(i, K_locs + r) for r in range(k)] for i in range(sim.kc)], []))]))

            # Broadcast K[vert][*] to all partitions
            for r in range(k):
                sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_ROW,
                    [sim.relToAbsCol(vert*k + r, K_unpacked_spread_start), sim.relToAbsCol(vert*k + r, K_unpacked_spread_start)],
                    [sim.relToAbsCol(0, K_locs + r)])]))
                log2_kc = sim.kc.bit_length() - 1
                for ll in range(log2_kc):
                    sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_ROW,
                        [sim.relToAbsCol(j, K_locs + r), sim.relToAbsCol(j, K_locs + r)],
                        [sim.relToAbsCol(j + (sim.kc >> (ll + 1)), K_locs + r)])
                        for j in range(0, sim.kc, 1 << (log2_kc - ll))]))

            # Compute the XNORs
            for r in range(k):
                XNOR(sim, overlap + a + r, K_locs + r, intermediates=intermediates)

            # Compute full-adder amongst the k=3 values
            assert(k == 3)  # Can be generalized
            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                [sim.relToAbsCol(partition, intermediates[0]), sim.relToAbsCol(partition, intermediates[1]),
            sim.relToAbsCol(partition, intermediates[2])]) for partition in range(sim.kc)]))

            sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
                [sim.relToAbsCol(j, K_locs)], [sim.relToAbsCol(j, intermediates[0])]) for j in range(sim.kc)]))
            sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                [sim.relToAbsCol(j, K_locs), sim.relToAbsCol(j, K_locs + 1), sim.relToAbsCol(j, K_locs + 2)],
                [sim.relToAbsCol(j, intermediates[1])]) for j in range(sim.kc)]))
            sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                [sim.relToAbsCol(j, K_locs + 2), sim.relToAbsCol(j, K_locs + 1), sim.relToAbsCol(j, intermediates[0])],
                [sim.relToAbsCol(j, intermediates[2])]) for j in range(sim.kc)]))

            sim.perform(ParallelOperation([Operation(GateType.INIT1, GateDirection.IN_ROW, [],
                [sim.relToAbsCol(partition, K_locs), sim.relToAbsCol(partition, K_locs + 1),
                ]) for partition in range(sim.kc)]))

            sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
                [sim.relToAbsCol(j, intermediates[1])], [sim.relToAbsCol(j, K_locs + 1)]) for j in range(sim.kc)]))
            sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                [sim.relToAbsCol(j, K_locs + 1), sim.relToAbsCol(j, intermediates[0]), sim.relToAbsCol(j, intermediates[2])],
                [sim.relToAbsCol(j, K_locs)]) for j in range(sim.kc)]))
            # K_locs stores the sum, K_locs + 1 stores the carry

            # In each partition, add the numbers represented by first_num (running_start) and second_num (K_locs)
            # As K_locs is only two bit, then we effectively sign-extend with zero
            first_num = running_start
            second_num = K_locs
            rep_size = ceil(log2(k ** 2))

            sim.perform(ParallelOperation(
                [Operation(GateType.INIT0, GateDirection.IN_ROW, [], [sim.relToAbsCol(j, K_locs + 2),
                                                                    sim.relToAbsCol(j, intermediates[0])])
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
                    [sim.relToAbsCol(j, first_num + k), sim.relToAbsCol(j, second_num + (k if k < 2 else 2)),
                     sim.relToAbsCol(j, carry_loc)], [sim.relToAbsCol(j, new_not_carry_loc)])
                    for j in range(sim.kc)]))
                sim.perform(ParallelOperation([Operation(GateType.NOT, GateDirection.IN_ROW,
                    [sim.relToAbsCol(j, new_not_carry_loc)], [sim.relToAbsCol(j, new_carry_loc)])
                    for j in range(sim.kc)]))
                sim.perform(ParallelOperation([Operation(GateType.MIN3, GateDirection.IN_ROW,
                    [sim.relToAbsCol(j, first_num + k), sim.relToAbsCol(j, second_num + (k if k < 2 else 2)),
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

            # Rotate A
            if vert < k-1:
                # mask = torch.LongTensor(sum([[sim.relToAbsCol(part, overlap + a) for a in range(k-1+na)] for part in range(sim.kc)], []))
                # sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN,
                #     [0, 0], [sim.r], mask)]))
                # sim.perform(ParallelOperation([Operation(GateType.OR, GateDirection.IN_COLUMN,
                #     [sim.relToAbsRow(part + 1, 0), sim.relToAbsRow(part + 1, 0)], [sim.r], mask) for part in range(0, sim.kr - 1, 2)]))
                for aaa in range(sim.kc):
                    sim.memory[:, sim.relToAbsCol(aaa, overlap):(sim.relToAbsCol(aaa, overlap)+k-1+na)] = \
                        torch.roll(sim.memory[:, sim.relToAbsCol(aaa, overlap):(sim.relToAbsCol(aaa, overlap)+k-1+na)], -1, dims=0)
                sim.latency += 32*4

        for aaa in range(sim.kc):
            sim.memory[:, sim.relToAbsCol(aaa, overlap):(sim.relToAbsCol(aaa, overlap)+k-1+na)] = \
                torch.roll(sim.memory[:, sim.relToAbsCol(aaa, overlap):(sim.relToAbsCol(aaa, overlap)+k-1+na)], k-1, dims=0)
        sim.latency += 32*4

        # Check if >= k // 2 + 1, in this case k >= 5
        # Check using (bit4 and (either bit1 or bit2)) or bit8
        OR(sim, running_start + 1, running_start)
        AND(sim, running_start, running_start + 2)
        OR(sim, running_start + 2, running_start + 3)
        Move(sim, running_start + 3, AK_start + a)
