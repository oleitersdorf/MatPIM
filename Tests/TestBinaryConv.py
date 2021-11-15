import torch
from Simulator.simulator import Simulator
from MatPIM.BinaryConv import BinaryConv


device = torch.device('cpu')


def testBinaryConv():
    """
    Tests the MatPIM binary convolution algorithm
    """

    # The parameters for the test
    r = 1024
    kr = 32
    c = 1024
    kc = 32
    m = 1024
    n = 256
    k = 3

    na = n // kc

    sim = Simulator([r // kr] * kr, [c // kc] * kc, device=device)

    print(f'Binary Convolution')
    print(f'Parameters: r={r}, c={c}, m={m}, n={n}, k={k}')

    # Construct random boolean matrix and kernel
    A = torch.rand(size=(m, n), device=device) < 0.5
    K = torch.rand(size=(k, k), device=device) < 0.5

    # Store the vectors in the memory
    for i in range(m):
        for j in range(n):
            sim.memory[i][sim.relToAbsCol(j // na, k - 1 + j % na)] = A[i][j]
    for j in range(k ** 2):
        sim.memory[j][sim.c] = K[j // k, j % k]

    # Run the matrix multiplication algorithm
    BinaryConv(sim, m, n, k)

    # Verify the results
    output = torch.zeros(size=(m - (k - 1), n), dtype=torch.bool, device=device)
    for i in range(m - (k - 1)):
        for j in range(n):
            output[i, j] = sim.memory[i][sim.relToAbsCol(j // na, k - 1 + na + j % na)]
    output = output[:, (k-1):]

    A = A.to(dtype=torch.int) * 2 - 1
    K = K.to(dtype=torch.int) * 2 - 1

    assert((output == (torch.nn.functional.conv2d(A.reshape(1, 1, m, n), K.reshape(1, 1, k, k)).squeeze(0).squeeze(0) >= 0)).all())
    print(f'Success with {sim.latency} cycles and {sim.energy} energy\n')


if __name__ == "__main__":
    testBinaryConv()
