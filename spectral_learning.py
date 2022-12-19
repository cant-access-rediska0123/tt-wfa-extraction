import numpy as np
from wfa_extraction import WFA
from scipy.sparse.linalg import svds


def spectral_learning(hPref, hSuf, H, rank, delta=1e-9):
    alphabet_size = H.shape[0] - 1    
    P = H.shape[1]
    S = H.shape[2]

    U, S, V = svds(H[0, :, :], k=rank, tol=delta)
    S_inv = np.zeros(rank, dtype=np.float32)
    for i in range(rank):
        if S[i] <= delta:
            S_inv[i] = 0
        else:
            S_inv[i] = 1 / S[i]

    U = U * S_inv
    alpha_0 = hSuf @ V.T
    alpha_inf = U.T @ hPref

    C = np.zeros(shape=(alphabet_size, rank, rank))
    for i in range(alphabet_size):
        C[i, :, :] = U.T @ H[i + 1, :, :] @ V.T

    return WFA(C, alpha_0, alpha_inf)
