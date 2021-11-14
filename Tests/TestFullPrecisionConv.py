import torch
from Simulator.simulator import Simulator
from MatPIM.FullPrecisionConv import FullPrecisionConv

N = 32

UINT_MIN = 0
UINT_MAX = 1 << N

device = torch.device('cpu')


def testFullPrecisionConv():
    """
    Tests the full-precision convolution algorithm
    """

    # The parameters for the test
    r = 1024
    c = 1024
    m = 1024
    n = 8
    k = 5
    alpha = n // 8

    sim = Simulator([1024], [32] * N, device=device)

    print(f'Full Precision 2D Convolution')
    print(f'Parameters: r={r}, c={c}, m={m}, n={n}, k={k}')

    # Construct random integer matrix and vector
    A = torch.randint(low=UINT_MIN, high=UINT_MAX, size=(m, n), device=device)
    K = torch.randint(low=UINT_MIN, high=UINT_MAX, size=(k, k), device=device)

    # Store the vectors in the memory
    for i in range(m):
        for j in range(n):
            sim.storeIntegerStrided(A[i, j].item(), 1 + k + (j % (n // alpha)), i + (j // (n // alpha)) * m)
    for j in range(k ** 2):
        sim.storeIntegerStrided(K[j // k, j % k].item(), 0, j)

    # Run the convolution algorithm
    FullPrecisionConv(sim, m, n, k, alpha)

    # Verify the results
    output = torch.zeros((m - (k - 1), n), dtype=torch.long, device=device)
    for i in range(m - (k - 1)):
        for j in range(n):
            output[i][j] = sim.loadIntegerStrided(1 + k + n//alpha + (j % (n // alpha)), i + (j // (n // alpha)) * m)
    if alpha > 1:
        output = output[:, (k-1):]
    else:
        output = output[:, :(n-(k-1))]

    assert((output == (torch.nn.functional.conv2d(A.reshape(1, 1, m, n), K.reshape(1, 1, k, k)) % (1 << N))).all())
    print(f'Success with {sim.latency} cycles and {sim.energy} energy\n')


if __name__ == "__main__":
    testFullPrecisionConv()
