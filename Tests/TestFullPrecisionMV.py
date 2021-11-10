import torch
from Simulator.simulator import Simulator
from MatPIM.FullPrecisionMV import FullPrecisionMV

N = 32

UINT_MIN = 0
UINT_MAX = 1 << N

device = torch.device('cpu')


def testFullPrecisionMV():
    """
    Tests the full-precision matrix-vector multiplication algorithm
    """

    # The parameters for the test
    r = 1024
    c = 1024
    alpha = 4
    m = 1024 // alpha
    n = 8 * alpha

    sim = Simulator([1024], [32] * N, device=device)

    print(f'Full Precision Matrix-Vector Multiplication')
    print(f'Parameters: r={r}, c={c}, m={m}, n={n}')

    # Construct random integer matrix and vector
    A = torch.randint(low=UINT_MIN, high=UINT_MAX, size=(m, n), device=device)
    x = torch.randint(low=UINT_MIN, high=UINT_MAX, size=(n, ), device=device)

    # Store the vectors in the memory
    for i in range(m):
        for j in range(n):
            sim.storeIntegerStrided(A[i, j].item(), j % (n // alpha), i + m * (j//(n//alpha)))
    for j in range(n):
        sim.storeIntegerStrided(x[j].item(), n//alpha + j % (n // alpha), m * (j//(n//alpha)))

    # Run the matrix multiplication algorithm
    FullPrecisionMV(sim, m, n, alpha)

    # Verify the results
    output = torch.zeros(m, dtype=torch.long, device=device)
    for i in range(m):
        output[i] = sim.loadIntegerStrided(2 * (n // alpha), i)

    assert((output == (torch.matmul(A, x) % (1 << N))).all())
    print(f'Success with {sim.latency} cycles and {sim.energy} energy\n')


if __name__ == "__main__":
    testFullPrecisionMV()
