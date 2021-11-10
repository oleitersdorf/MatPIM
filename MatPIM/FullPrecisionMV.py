from MatPIM.Utilities import *


def FullPrecisionMV(sim: Simulator, m: int, n: int, alpha: int):
    """
    Performs the MatPIM full-precision matrix-vector multiplication algorithm.
    :param sim: the simulation environment
    :param m: the number of rows in the matrix
    :param n: the number of columns in the matrix
    :param alpha: the number of blocks
    """

    # The effective number of elements per row from the vector
    na = n // alpha

    # Clone x along rows
    for a in range(alpha):
        VBroadcast(sim, a * m, list(range(a * m, a * m + m)), list(range(na, 2*na)))

    # Perform inner product in parallel
    InnerProduct(sim, na, list(range(na)), list(range(na, 2*na)), 2*na, list(range(alpha * m)))

    # Perform reduction
    a = alpha
    while a > 1:

        blocks_to_shift = range(alpha//a, alpha, 2*alpha//a)
        blocks_to_receive = range(0, alpha, 2*alpha//a)

        # Shift right
        Move(sim, 2*na, 2*na + 1, mask=list(range(alpha * m)))

        # Perform VCOPY
        VCOPY(sim, sum([list(range(start*m, start*m+m)) for start in blocks_to_shift], []), sum([list(range(start*m, start*m+m)) for start in blocks_to_receive], []), [2*na + 1])

        # Perform addition
        Add(sim, 2*na, 2*na + 1, 2*na, mask=list(range(alpha * m)))

        a //= 2
