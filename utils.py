def absolute_or_relative_error(f1, f2):
    return np.abs(f1 - f2) / np.maximum(np.maximum(np.abs(f1), np.abs(f2)), np.ones(f1.shape[0]))


def make_equal_length(tests):
    max_len = 1
    tests_len = len(tests)
    for i in range(tests_len):
        max_len = max(max_len, len(tests[i]))
    for i in range(tests_len):
        need = max_len - len(tests[i])
        for it in range(need):
            tests[i].append(0)


def get_function_values(W, tests):
    result = []
    for seq in tests:
        result.append(W.f(seq))
    return np.array(result)


def func_difference_metrics(W1, W2, max_length, max_iter=1000):
    arr1 = np.zeros(max_length + 1)
    arr2 = np.zeros(max_length + 1)

    test_all = True
    for l in range(max_length + 1):
        tests = []
        if test_all and W1.alphabet_size ** l <= max_iter:
            domain = [list(range(1, W1.alphabet_size + 1, 1)) for i in range(l)]
            seqs = itertools.product(*domain)
            for seq in seqs:
                if l == 0:
                    tests.append([0])
                else:
                    tests.append(list(seq))
        else:
            test_all = False
            for iter in range(max_iter):
                seq = np.random.randint(W1.alphabet_size, size=l) + 1
                tests.append(seq)

        ans1 = get_function_values(W1, tests)
        ans2 = get_function_values(W2, tests)
        diffs = absolute_or_relative_error(ans1, ans2)
        arr1[l] = np.amax(diffs)
        arr2[l] = np.mean(diffs)

    return arr1, arr2


import numpy as np
import itertools


def make_set_of_strings(alphabet_size, len=-1, limit=250):
    if len == -1:
        res = [[]]
        limit -= 1
        for l in range(1, limit + 1, 1):
            count = alphabet_size ** l
            if count <= limit:
                domain = [list(range(1, alphabet_size + 1, 1)) for i in range(l)]
                seqs = itertools.product(*domain)
                for seq in seqs:
                    res.append(list(seq))
                limit -= count
            else:
                for iter in range(limit):
                    seq = np.random.randint(alphabet_size, size=l) + 1
                    res.append(list(seq))
                break
        return res

    res = [[]]
    if len == 0:
        return res
    limit -= 1

    up = limit // len
    rem = limit % len
    for l in range(1, len + 1, 1):
        count = alphabet_size ** l
        if count <= up + rem:
            domain = [list(range(1, alphabet_size + 1, 1)) for i in range(l)]
            seqs = itertools.product(*domain)
            for seq in seqs:
                res.append(list(seq))
            limit -= count
            if l < len:
                up = limit // (len - l)
                rem = limit % (len - l)
        else:
            for iter in range(up + rem):
                seq = np.random.randint(alphabet_size, size=l) + 1
                res.append(list(seq))
            rem = 0
    return res


def make_set_of_random_strings(alphabet_size, len, size=200):
    res = []
    for i in range(size):
        seq = np.random.randint(alphabet_size, size=len) + 1
        res.append(seq)
    return res
