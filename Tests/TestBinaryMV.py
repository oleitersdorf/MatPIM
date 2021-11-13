import torch
from Simulator.simulator import Simulator
from MatPIM.BinaryMV import BinaryMV
from math import ceil, log2

device = torch.device('cpu')


def testBinaryMV():
    """
    Tests the binary matrix-vector multiplication algorithm
    """

    # The parameters for the test
    r = 1024
    kr = 32
    c = 1024
    kc = 32
    m = 1024
    n = 384

    np = n // (c // kc)

    sim = Simulator([r // kr] * kr, [c // kc] * kc, device=device)

    print(f'Binary Matrix-Vector Multiplication')
    print(f'Parameters: r={r}, c={c}, m={m}, n={n}')

    # Construct random boolean matrix and vector
    A = torch.rand(size=(m, n), device=device) < 0.5
    x = torch.rand(size=(n, ), device=device) < 0.5

    # Store the vectors in the memory
    for i in range(m):
        for j in range(n):
            sim.memory[i][sim.relToAbsCol(j // np, j % np)] = A[i][j]
    for j in range(n):
        sim.memory[0][sim.relToAbsCol(j // np, np + j % np)] = x[j]

    # Run the matrix multiplication algorithm
    BinaryMV(sim, m, n)

    # Verify the results
    output = torch.zeros(m, dtype=torch.int, device=device)
    for i in range(m):
        output[i] = sum([int(sim.memory[i][np + r]) << r for r in range(ceil(log2(n)))])

    A = A.to(dtype=torch.int) * 2 - 1
    x = x.to(dtype=torch.int) * 2 - 1

    assert((output == (torch.matmul(A, x) // 2 + n // 2)).all())
    print(f'Success with {sim.latency} cycles and {sim.energy} energy\n')


if __name__ == "__main__":
    testBinaryMV()
