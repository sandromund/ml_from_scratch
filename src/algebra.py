from math import sqrt


def matrix_transpose(A):
    return [[A[j][i] for j in range(len(A))]
            for i in range(len(A[0]))]


def matrix_subtraktion(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def matrix_multiplication(A, B):
    result = []
    for k in range(len(A)):
        k_result = []
        for i in range(len(B[0])):
            sum_i = 0
            for j in range(len(A[0])):
                sum_i += (A[k][j] * B[j][i])
            k_result += [sum_i]
        result += [k_result]
    return result


def dot_product(v, w):
    return sum([v[i] * w[i] for i in range(len(v))])


def l1_norm(v):
    return sum([abs(v[i]) for i in range(len(v))])


def l2_norm(v):
    return sqrt(sum([v[i] ** 2 for i in range(len(v))]))
