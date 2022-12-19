import functools

import tntorch as tn
import torch


def wfa_function(vec, W):
    res = torch.zeros(vec.shape[0])
    for i in range(vec.shape[0]):
        res[i] = W.f(vec[i].int())
    return res


def extract_tensor_with_cross(W, n, rank, max_iter=25, eps_cross=1e-6, return_info=False, kickrank=0, rmax=-1):
    domain = [torch.arange(W.alphabet_size + 1, dtype=int) for i in range(2 * n + 1)]

    func = functools.partial(wfa_function, W=W)

    if rmax == -1:
        rmax = rank

    if kickrank == 0:
        return tn.cross(function=func,
                        domain=domain,
                        function_arg='matrix',
                        ranks_tt=rank,
                        max_iter=max_iter,
                        eps=eps_cross,
                        kickrank=0,
                        rmax=rmax,
                        return_info=return_info)

    return tn.cross(function=func,
                    domain=domain,
                    function_arg='matrix',
                    max_iter=max_iter,
                    eps=eps_cross,
                    kickrank=kickrank,
                    rmax=rmax,
                    return_info=return_info)
