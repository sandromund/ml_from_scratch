from functools import reduce
from itertools import combinations


def merge_subsets(M):
    result = []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            result += [M[i] | M[j]]
    return set(result)


def support_filter(c_i, transactions, threshold=0.5):
    return [frozenset(c) for c in c_i if support(c, transactions) >= threshold]


def get_disjoint_subsets(M):
    return [(x, M - x) for x in [set(x) for x in list(combinations(M, len(M) - 1))]]


def support(c, A):
    return [set(c).issubset(a_i) for a_i in A].count(True) / len(X)


def confidence(L, R, M):
    return (support(L | R, M)) / support(L, M)


def apriori_algorithm(transactions, min_support, min_confidence):
    # step 1: select all candidates
    c_i = [frozenset(x) for x in
           reduce(lambda i, j: i | j, transactions)]
    candidates = []
    while c_i:
        c_i = merge_subsets(c_i)
        c_i = support_filter(c_i, transactions, threshold=min_support)
        candidates += c_i

    # step 2:
    for candidat in candidates:
        for L, R in get_disjoint_subsets(candidat):
            conv = round(confidence(L, R, transactions), 3)
            if conv >= min_confidence:
                print(L, '=>', set(R), 'with ', conv, ' confidence')


if __name__ == '__main__':

    C = ['A', 'B', 'C', 'D', 'E']

    X = [[1, 1, 1, 0, 1],
         [0, 0, 1, 1, 0],
         [1, 1, 0, 0, 1],
         [1, 0, 1, 0, 0],
         [1, 1, 1, 1, 1]]

    transactions = [frozenset([C[j] for j in range(len(C)) if X[i][j] == 1])
                    for i in range(len(X))]

    apriori_algorithm(transactions, min_support=0.6, min_confidence=0.8)