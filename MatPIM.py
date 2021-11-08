import torch
from simulator import Simulator, ParallelOperation, Operation, GateType, GateDirection


def FullPrecisionMV(sim: Simulator, m: int, n: int):

    # Clone x along rows
    x = torch.zeros(size=(n,), dtype=torch.long, device=sim.device)
    for j in range(n):
        x[j] = sim.loadIntegerStrided(n + j, 0)
    for i in range(m):
        for j in range(n):
            sim.storeIntegerStrided(x[j], n + j, i)

    # Perform inner product in parallel
    for i in range(m):

        my_a = torch.zeros(size=(n,), dtype=torch.long, device=sim.device)
        my_x = torch.zeros(size=(n,), dtype=torch.long, device=sim.device)
        for j in range(n):
            my_a[j] = sim.loadIntegerStrided(j, i)
            my_x[j] = sim.loadIntegerStrided(n + j, i)

        my_ax = torch.dot(my_a, my_x)

        sim.storeIntegerStrided(my_ax.item(), 2 * n, i)
