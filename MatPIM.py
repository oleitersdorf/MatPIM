import torch
from typing import List
from simulator import Simulator, ParallelOperation, Operation, GateType, GateDirection


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


def FullPrecisionMV(sim: Simulator, m: int, n: int):
    """
    Performs the MatPIM full-precision matrix-vector multiplication algorithm.
    :param sim: the simulation environment
    :param m: the number of rows in the matrix
    :param n: the number of columns in the matrix
    :return:
    """

    # Clone x along rows
    x = torch.zeros(size=(n,), dtype=torch.long, device=sim.device)
    for j in range(n):
        x[j] = sim.loadIntegerStrided(n + j, 0)
    for i in range(m):
        for j in range(n):
            sim.storeIntegerStrided(x[j], n + j, i)

    # Perform inner product in parallel
    InnerProduct(sim, n, list(range(n)), list(range(n, 2*n)), 2*n, list(range(m)))
