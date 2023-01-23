from itertools import combinations
import numpy as np


def SIMPLEX_Bland(L_B, print_log=False):
    B, N, A, c, b, v = L_B

    n = len(A[0])
    m = len(b)
    get_cols = lambda A, B: [[A[row][i] for i in B] for row in range(len(A))]

    if print_log:
        print("A: ", A)
        print("B: ", B, " N:", N)
        print("c: ", c)
        print("-" * 100)

    while True:

        # --------------------- Schritt 1 ---------------------

        # Wähle e Element von N mit C_e > 0
        # also beitrettende Variable ist x_e
        c_e = [c[i] for i in N]
        # Falls kein solches e exisitiert,
        #  dann STOP mit aktueller Slackform zu zughöriger Basislösung
        if not any(map(lambda x: x > 0, c_e)):
            print("STOP")
            return B, N, A, c, b, v
        else:
            # nehme bei mehreren Möglichkeiten für x_e die Variable mit kleinesten Index.
            max_index, max_value = -1, -1
            for i in range(len(c)):
                if c[i] > max_value and i in N:
                    max_index, max_value = i, c[i]
            e = max_index  # e = c.index(max(c))

        #  --------------------- Schritt 2 ---------------------

        # Falls für alle i Element von B gilt, a_ie <= 0,
        #  dann STOP mit Meldung UNBESCHRÄNKT
        if all(map(lambda i: A[i][e] <= 0, [i for i in range(len(B))])):
            print("UNBESCHRÄNKT")
            return B, N, A, c, b, v

        else:
            # sonst wähle l Element von B wobei a_ie > 0 ist
            # so dass b_l / a_le minimal ist
            # bei mehren Möglichkeiten für x_l die Variante mit kleinstem Index wählen
            l = min([l for l in range(m) if A[l][e] > 0], key=lambda l: b[l][0] / A[l][e])
            l = B[l]

        #  --------------------- Schritt 3 ---------------------

        # Konstruiere neue zulässige Slackform L(B)

        bs = np.dot(np.linalg.inv(get_cols(A, B)), b)  # b' = A_B^-1 * b
        As_B_inv = np.linalg.inv(get_cols(A, B))  # A'_B^-1
        As = np.dot(As_B_inv, get_cols(A, N))
        c_B = [c[i] for i in B]
        v = np.dot(np.dot(As_B_inv, c_B), b)[0]
        c_N = [c[i] for i in N]
        c_ = np.subtract(c_N, np.dot(np.dot(c_B, As_B_inv), get_cols(A, N)))

        # da c_ nur die N Vars. enthält, werden die in der Iteration nicht betroffenen Spalten wieder eingebaut

        k = 0
        c_new = []
        for i in range(n):
            if i == l:
                c_new += [0]

            elif i not in N:
                c_new += [c[i]]
            else:
                c_new += [c_[k]]
                k += 1

        A_new = []
        for j in range(m):
            k = 0
            A_j = []
            for i in range(n):
                if i not in N:
                    A_j += [A[j][i]]
                else:
                    A_j += [As[j][k]]
                    k += 1
            A_new += [A_j]

        A = A_new
        b = bs
        c = c_new

        # z = v + np.dot(c_, X_N)
        #  --------------------- Schritt 4 ---------------------

        N.remove(e)
        N.append(l)
        N.sort()

        B.remove(l)
        B.append(e)
        B.sort()

        if print_log:
            print("A: ", A)
            print("b: ", b)
            print("B: ", B)
            print("N: ", N)
            print("c: ", c_new)
            print("v: ", v)
            print("-" * 100)


if __name__ == '__main__':
    c = [19, 13, 12, 17]

    A = [[3, 2, 1, 2],
         [1, 1, 1, 1],
         [4, 3, 3, 4]]

    b = [[225],
         [117],
         [420]]

    n = len(A[0])
    m = len(b)
    N = list(range(n))
    B = list(range(n, n + len(b)))

    # A' konstruieren
    A_ = A
    for i in range(m):
        v = [0] * m
        v[i] = 1
        A_[i] += v

    # c' konstruieren
    c += [0] * m

    # init
    v = 0

    L_1_8 = (B, N, A, c, b, v)
    SIMPLEX_Bland(L_1_8, print_log=True)
