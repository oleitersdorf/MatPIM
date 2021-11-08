import torch
from typing import List
from enum import Enum


class GateType(Enum):
    """
    Represents a type of gate out of the supported stateful gates.
    """

    NOT = 0
    NOR = 1
    NAND = 2
    OR = 3
    MIN3 = 4
    INIT0 = 5
    INIT1 = 6


class GateDirection(Enum):
    """
    Represents the direction of the gate
    """

    IN_ROW = 0
    IN_COLUMN = 1


class Operation:
    """
    Represent a single row/column operation
    """

    def __init__(self, gateType: GateType, gateDirection: GateDirection, inputs: List[int], outputs: List[int],
                 mask: torch.LongTensor = None):
        """
        Constructs an operation object
        :param gateType: the type of gate (e.g., NOR)
        :param gateDirection: the direction of the gate (e.g., IN_ROW)
        :param inputs: the absolute input addresses of the row/columns
        :param outputs: the absolute output addresses of the row/columns
        :param mask: the mask on the parallel operation (non-isolated gates)
        """
        self.gateType = gateType
        self.gateDirection = gateDirection
        self.inputs = inputs
        self.outputs = outputs
        self.mask = mask

    def getEnergy(self):
        """
        Returns the energy (gate count) in the overall operation
        :return:
        """
        if self.mask is None:
            return r if (self.gateDirection == GateDirection.IN_ROW) else c
        else:
            return len(self.mask)


class ParallelOperation:
    """
    Represent multiple single row/column operations
    """

    def __init__(self, ops: List[Operation]):
        """
        Constructs a parallel operation object
        :param gates: the set of column ops that construct the overall operation
        """
        self.ops = ops
        # TODO: Check collision


class Simulator:
    """
    Simulates a single crossbar that supports stateful logic with partitions along both dimensions
    """

    def __init__(self, row_partition_sizes: List[int], col_partition_sizes: List[int], device: torch.device):
        """
        Initializes the simulator according to the partition sizes, constructing a grid of partitions
        :param row_partition_sizes: A list containing the size of each partition in each row
        :param col_partition_sizes: A list containing the size of each partition in each column
        :param device: The device (e.g., CPU, GPU) to utilize
        """

        # Initialize the memory
        self.r = sum(row_partition_sizes)
        self.c = sum(col_partition_sizes)
        self.device = device
        self.memory = torch.zeros(self.r, self.c, dtype=torch.bool, device=device)

        # Initialize the partition address translation
        self.kr = len(row_partition_sizes)
        self.kc = len(col_partition_sizes)
        self.row_partition_starts = [sum(row_partition_sizes[:i]) for i in range(self.kr)]
        self.col_partition_starts = [sum(col_partition_sizes[:i]) for i in range(self.kc)]

        # Initialize the counters
        self.latency = 0
        self.energy = 0

    def relToAbsRow(self, partition, index):
        """
        Converts a row address from (partition, intra-partition index) to a global address
        :param partition: the row partition
        :param index: the intra-partition index
        :return: an absolute row address
        """
        return self.row_partition_starts[partition] + index

    def relToAbsCol(self, partition, index):
        """
        Converts a column address from (partition, intra-partition index) to a global address
        :param partition: the column partition
        :param index: the intra-partition index
        :return: an absolute column address
        """
        return self.col_partition_starts[partition] + index

    def perform(self, parallelOp: ParallelOperation):
        """
        Performs the given parallel operation on the simulation crossbar
        :param parallelOp: the parallel operation to perform
        """

        for op in parallelOp.ops:
            performOperation(op)
            self.energy += op.getEnergy()
        self.latency += 1

    def performOperation(self, operation: Operation):
        """
        Performs a single operation on the crossbar
        :param operation: the operation to perform
        """

        mask = operation.mask if operation.mask else list(range(self.r if operation.gateDirection == GateDirection.IN_ROW else self.c))

        if operation.gateType == GateType.NOT:

            if operation.gateDirection == GateDirection.IN_ROW:
                self.memory[mask, operation.outputs[0]] = torch.bitwise_and(self.memory[mask, operation.outputs[0]],
                torch.bitwise_not(self.memory[mask, operation.inputs[0]]))
            else:
                self.memory[operation.outputs[0], mask] = torch.bitwise_and(self.memory[operation.outputs[0], mask],
                torch.bitwise_not(self.memory[operation.inputs[0], mask]))

        elif operation.gateType == GateType.NOR:

            if operation.gateDirection == GateDirection.IN_ROW:
                self.memory[mask, operation.outputs[0]] = torch.bitwise_and(self.memory[mask, operation.outputs[0]],
                torch.bitwise_nor(self.memory[mask, operation.inputs[0]], self.memory[mask, operation.inputs[1]]))
            else:
                self.memory[operation.outputs[0], mask] = torch.bitwise_and(self.memory[operation.outputs[0], mask],
                torch.bitwise_nor(self.memory[operation.inputs[0], mask], self.memory[operation.inputs[1], mask]))

        elif operation.gateType == GateType.NAND:

            if operation.gateDirection == GateDirection.IN_ROW:
                self.memory[mask, operation.outputs[0]] = torch.bitwise_and(self.memory[mask, operation.outputs[0]],
                torch.bitwise_not(torch.bitwise_and(self.memory[mask, operation.inputs[0]], self.memory[mask, operation.inputs[1]])))
            else:
                self.memory[operation.outputs[0], mask] = torch.bitwise_and(self.memory[operation.outputs[0], mask],
                torch.bitwise_not(torch.bitwise_and(self.memory[operation.inputs[0], mask], self.memory[operation.inputs[1], mask])))

        elif operation.gateType == GateType.OR:

            if operation.gateDirection == GateDirection.IN_ROW:
                self.memory[mask, operation.outputs[0]] = torch.bitwise_and(self.memory[mask, operation.outputs[0]],
                torch.bitwise_or(self.memory[mask, operation.inputs[0]], self.memory[mask, operation.inputs[1]]))
            else:
                self.memory[operation.outputs[0], mask] = torch.bitwise_and(self.memory[operation.outputs[0], mask],
                torch.bitwise_or(self.memory[operation.inputs[0], mask], self.memory[operation.inputs[1], mask]))

        elif operation.gateType == GateType.MIN3:

            if operation.gateDirection == GateDirection.IN_ROW:
                self.memory[mask, operation.outputs[0]] = torch.bitwise_and(self.memory[mask, operation.outputs[0]],
                torch.bitwise_not(torch.bitwise_or(torch.bitwise_and(self.memory[mask, operation.inputs[0]], self.memory[mask, operation.inputs[1]]),
                                     torch.bitwise_or(
                                         torch.bitwise_and(self.memory[mask, operation.inputs[0]], self.memory[mask, operation.inputs[1]]),
                                         torch.bitwise_and(self.memory[mask, operation.inputs[0]], self.memory[mask, operation.inputs[1]])
                                     ))))
            else:
                self.memory[operation.outputs[0], mask] = torch.bitwise_and(self.memory[operation.outputs[0], mask],
                torch.bitwise_nor(self.memory[operation.inputs[0], mask], self.memory[operation.inputs[1], mask]))

        elif operation.gateType == GateType.INIT0:

            for output in operation.outputs:

                if operation.gateDirection == GateDirection.IN_ROW:
                    self.memory[mask, output] = False
                else:
                    self.memory[output, mask] = False

        elif operation.gateType == GateType.INIT1:

            for output in operation.outputs:

                if operation.gateDirection == GateDirection.IN_ROW:
                    self.memory[mask, output] = True
                else:
                    self.memory[output, mask] = True

    def storeIntegerStrided(self, x: int, loc: int, row: int):
        """
        -- Only to be used by the testing environment --
        Writes the integer x in a strided representation with intra-partition index loc (horizontally), in the given row.
        Note: MSB stored in partition 0
        :param x: the integer (kr-bit)
        :param loc: the intra-partition index (horizontal)
        :param row: the row containing the number
        """

        for i in range(self.kc):
            self.memory[row][self.relToAbsCol(i, loc)] = bool(x & (1 << (self.kc - i - 1)))

    def loadIntegerStrided(self, loc: int, row: int):
        """
        -- Only to be used by the testing environment --
        Reads the integer x in a strided representation with intra-partition index loc (horizontally), from the given row.
        Note: MSB stored in partition 0
        :param loc: the intra-partition index (horizontal)
        :param row: the row containing the number
        """

        x = 0

        for i in range(self.kc):
            x += self.memory[row][self.relToAbsCol(i, loc)].item() << (self.kc - i - 1)

        return x