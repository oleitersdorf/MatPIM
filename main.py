import time
import torch
from tqdm import tqdm
from simulator import Simulator
from MatPIM import FullPrecisionMV

UINT_MIN = 0
UINT_MAX = 1 << 32

device = torch.device('cpu')


def testFullPrecisionMV():
    """
    Tests the full-precision matrix-vector multiplication algorithm
    """

    # The parameters for the test
    r = 1024
    c = 1024
    m = 1024
    n = 8

    sim = Simulator([1024], [32] * 32, device=device)

    print(f'Full Precision Matrix-Vector Multiplication')
    print(f'Parameters: r={r}, c={c}, m={m}, n={n}')

    # Construct random integer matrix and vector
    A = torch.randint(low=UINT_MIN, high=UINT_MAX, size=(m, n), device=device)
    x = torch.randint(low=UINT_MIN, high=UINT_MAX, size=(n, ), device=device)

    # Store the vectors in the memory
    for i in range(m):
        for j in range(n):
            sim.storeIntegerStrided(A[i, j].item(), j, i)
    for j in range(n):
        sim.storeIntegerStrided(x[j].item(), n + j, 0)

    # Run the matrix multiplication algorithm
    FullPrecisionMV(sim, m, n)

    # Verify the results
    output = torch.zeros(m, dtype=torch.long, device=device)
    for i in range(m):
        output[i] = sim.loadIntegerStrided(2 * n, i)

    assert((output == (torch.matmul(A, x) % (1 << 32))).all())
    print(f'Success with {sim.latency} cycles and {sim.energy} energy\n')


if __name__ == "__main__":
    testFullPrecisionMV()
