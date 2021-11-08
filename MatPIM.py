import torch
from simulator import Simulator, ParallelOperation, Operation, GateType, GateDirection


def FullPrecisionMV(sim: Simulator, m: int, n: int):

    # Construct random integer matrix and vector
    A = torch.zeros(size=(m, n), dtype=torch.long, device=sim.device)
    x = torch.zeros(size=(n,), dtype=torch.long, device=sim.device)

    # Store the vectors in the memory
    for i in range(m):
        for j in range(n):
            A[i, j] = sim.loadIntegerStrided(j, i)
    for j in range(n):
        x[j] = sim.loadIntegerStrided(n + j, 0)

    Ax = torch.matmul(A, x)
    for i in range(m):
        sim.storeIntegerStrided(Ax[i].item(), 2 * n, i)
