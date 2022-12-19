import torch
from utils import get_function_values, make_set_of_strings, make_set_of_random_strings

import numpy as np


class WFA:
    def __init__(self, A, alpha_0, alpha_inf):
        self.alphabet_size = A.shape[0]
        self.rank = A.shape[1]

        self.A = A
        self.alpha_0 = alpha_0
        self.alpha_inf = alpha_inf

    def multiply(self, c):
        self.alpha_0 *= c

    def f(self, s):
        v = self.alpha_0
        for sigma in s:
            if sigma > 0 and sigma <= self.alphabet_size:
                v = (v @ self.A[sigma - 1, :, :])
        return v @ self.alpha_inf

    def __repr__(self):
        answer = ''
        answer += "WFA (alphabet_size={}, rank={}):\n".format(self.alphabet_size, self.rank)
        answer += "alpha_0: {}\n".format(self.alpha_0)
        for i in range(self.alphabet_size):
            answer += "A_{}:\n{}\n".format(i + 1, self.A[i, :, :])
        answer += "alpha_inf: {}\n".format(self.alpha_inf)
        return answer

    def __str__(self):
        return self.__repr__()


class RandomNormalWFA(WFA):
    def __init__(self, alphabet_size, rank, l=1, coef=1, seed=None, lval=1, rval=1):
        if seed is not None:
            np.random.seed(seed)

        l = np.sqrt(l)

        alpha_0 = np.random.uniform(size=rank, low=-1, high=1)
        alpha_0 /= np.linalg.norm(alpha_0)
        alpha_0 *= l

        alpha_inf = np.random.uniform(size=rank, low=-1, high=1)
        alpha_inf /= np.linalg.norm(alpha_inf)
        alpha_inf *= l

        mlt = np.random.uniform(size=alphabet_size, low=lval, high=rval)
        A = np.random.uniform(size=(alphabet_size, rank, rank), low=-1, high=1)
        for sigma in range(alphabet_size):
            Q, R = np.linalg.qr(A[sigma, :, :], mode='complete')
            A[sigma, :, :] = (Q * coef) * mlt[sigma]

        super().__init__(A, alpha_0, alpha_inf)


def wfa_extraction(W, alphabet_size=None, rank=None, delta=1e-6):
    W = W.clone()

    N = (W.dim() - 1) // 2

    if alphabet_size is None:
        alphabet_size = W.size()[0] - 1

    if rank is None:
        rank = W.ranks_tt[N]

    for i in range(N):
        W.left_orthogonalize(i)
    for i in range(N):
        W.right_orthogonalize(2 * N - i)

    U, S, V = torch.svd(W.cores[N][:, 0, :].double())
    s = len(S)
    for i in range(len(S)):
        if S[i] <= delta:
            s = i
            break
    S_inv = np.zeros(len(S), dtype=np.float64)
    S_inv[0:s] = 1 / S[0:s]
    U = U * S_inv

    U = U.float()
    V = V.float()

    if U.shape[1] > rank:
        U = U[:, :rank]
    elif U.shape[1] < rank:
        U = torch.hstack((U, torch.zeros((U.shape[0], rank - U.shape[1]))))

    if V.shape[1] > rank:
        V = V[:, :rank]
    elif V.shape[1] < rank:
        V = torch.hstack((V, torch.zeros((V.shape[0], rank - V.shape[1]))))

    H = np.zeros(shape=(alphabet_size, rank, rank))
    for i in range(alphabet_size):
        H[i, :, :] = U.T @ W.cores[N][:, i + 1, :] @ V

    alpha_0 = W.cores[0][0][0]
    for i in range(N):
        alpha_0 = alpha_0 @ W.cores[i + 1][:, 0, :]
    alpha_0 = alpha_0 @ V

    alpha_inf = W.cores[2 * N].T[0][0]
    for i in range(N):
        alpha_inf = W.cores[2 * N - 1 - i][:, 0, :] @ alpha_inf
    alpha_inf = U.T @ alpha_inf

    return WFA(H, np.array(alpha_0.double()), np.array(alpha_inf.double()))


def create_hankel_matrices(W: WFA, lPref=-1, lSuf=-1, kPref=200, kSuf=200):
    alphabet_size = W.alphabet_size

    Pref = make_set_of_strings(alphabet_size, lPref, kPref)
    P = len(Pref)

    Suf = make_set_of_strings(alphabet_size, lSuf, kSuf)
    S = len(Suf)

    hPref = get_function_values(W, Pref)
    hSuf = get_function_values(W, Suf)

    H = np.zeros((alphabet_size + 1, P, S))
    for c in range(alphabet_size + 1):
        tests = []
        for i in range(P):
            for j in range(S):
                tests.append(Pref[i] + [c] + Suf[j])
        H[c, :, :] = np.reshape(get_function_values(W, tests), (P, S))

    return hPref, hSuf, H


def create_hankel_matrices_on_random_sets(W: WFA, lPref, lSuf, kPref=200, kSuf=200):
    alphabet_size = W.alphabet_size

    Pref = make_set_of_random_strings(alphabet_size, lPref, kPref)
    P = len(Pref)

    Suf = make_set_of_random_strings(alphabet_size, lSuf, kSuf)
    S = len(Suf)

    hPref = get_function_values(W, Pref)
    hSuf = get_function_values(W, Suf)

    H = np.zeros((alphabet_size + 1, P, S))
    for c in range(alphabet_size + 1):
        tests = []
        for i in range(P):
            for j in range(S):
                tests.append(list(Pref[i]) + [c] + list(Suf[j]))
        H[c, :, :] = np.reshape(get_function_values(W, tests), (P, S))

    return hPref, hSuf, H
