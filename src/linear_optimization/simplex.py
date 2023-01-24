import numpy as np


class SIMPLEX:

    def __init__(self, L, L_B):
        # self.L = L
        # self.L_B = L_B

        self.b = L.b
        self.B = L_B.B
        self.cq = L_B.cq
        self.Aq = L_B.Aq
        self.bq = L_B.bq

        self.A_ = np.append(L.A, np.identity(L.m), axis=1)
        self.c_ = np.append(L.c, np.zeros(L.m))

        self.n = self.A_.shape[1]
        self.m = self.A_.shape[0]

        # STEP 0
        self.N = set(range(self.n)) - self.B

        self.terminated = False

        self.e = None
        self.l = None
        self.L_B = None

        self.name = L.name

        self.start_str = "=== Starte SIMPLEX ==="
        self.end_str = "=== Ende SIMPLEX ===\n"

        self.debug_print = True
        self.n_spaces = 3

        print(self.info())

    def info(self):
        s = "SIMPLEX: %s \n"
        s += "A': \n %s \n"
        s += "c': \n %s \n"
        s += "N: \n %s \n"
        s += "B: \n %s \n"
        return s % (self.name, self.A_, self.c_, self.N, self.B)

    def run(self, n_iterations=None):
        print(self.start_str)
        iteration_index = 0

        while not self.terminated:
            if self.debug_print:
                print(" " * self.n_spaces + "-> Iteration " + str(iteration_index + 1))
            if n_iterations is not None and iteration_index >= n_iterations:
                print("STOP - maximale interation erreicht")
                print(self.end_str)
                return self.L_B
            else:
                self.step_1()
                self.step_2()
                self.step_3()
                iteration_index += 1
                print()

    def step_1(self):
        if self.terminated:
            return

        if self.debug_print:
            print(" " * self.n_spaces * 2 + "Step 1:")
            # print(" "*self.n_spaces*3, "cq: ", self.cq)

        if all(self.cq <= 0):
            print(" " * self.n_spaces * 3, "--> STOP mit aktueller Slackform - optimaler Lösung")
            print(self.end_str)
            self.terminated = True
            return self.L_B
        else:
            # wähle e € N mit cq_e > 0  und kleinseten Index
            n = self.cq.shape[0]
            possible_e = [i[0] for i in list(zip(range(n), self.c_)) if i[0] in self.N and i[1] > 0]
            self.e = min(possible_e)

            if self.debug_print:
                print(" " * self.n_spaces * 3, "-> beitetenden Variabel ist x_" + str(self.e))

    def step_2(self):
        if self.terminated:
            return

        if self.debug_print:
            print(" " * self.n_spaces * 2 + "Step 2:")

        if all(self.Aq[:, self.e] <= 0):
            print("STOP -> UNBESCHRÄNKT")
            print(self.end_str)
            self.terminated = True
            return None
        else:
            # wähle l mit l € B mit Aq_ie <= 0 so dass bq_l / Aq_le minimal mit kleinstem index
            v_min = None
            l = None
            b_index = None

            # print(self.Aq)

            for i, B_i in zip(range(len(self.B)), self.B):
                # print(self.B, self.L_B.Aq, i-n, self.e)
                if self.Aq[i][self.e] > 0:
                    v = self.bq[i] / self.Aq[i][self.e]

                    if (v_min is None) or (v < v_min) or (v == v_min and B_i < b_index):
                        l = self.Aq[i].shape[0] + i
                        v_min = v
                        b_index = B_i

                    if self.debug_print:
                        s = " " * self.n_spaces * 3
                        s += "bq_" + str(i) + " : Aq_" + str(i) + str(self.e) + " = " + str(round(v[0], 3))
                        s += " = " + str(self.bq[i][0]) + " / " + str(self.Aq[i][self.e])
                        s += " mit l = " + str(B_i)
                        print(s)
            self.l = b_index
            if self.debug_print:
                print(" " * self.n_spaces * 3, "-> verlassende Variabel ist x_" + str(b_index))

    def step_3(self):
        if self.terminated:
            return

        if self.debug_print:
            print(" " * self.n_spaces * 2 + "Step 3:")
        self.B -= {self.l}
        self.B.add(self.e)

        self.N = self.N - {self.e}
        self.N.add(self.l)

        print(" " * self.n_spaces * 3 + "B: " + str(self.B))
        print(" " * self.n_spaces * 3 + "N: " + str(self.N))

        # Konstruiere L(B)
        A_ = self.A_
        b = self.b
        c_ = self.c_

        B = self.B
        N = self.N
        B = sorted(list(B))
        N = sorted(list(N))

        A_B_inv = np.linalg.inv(A_[:, B])
        bq = A_B_inv @ b
        Aq = A_B_inv @ A_[:, N]
        x_N = np.array([0 for _ in sorted(list(N))]).reshape(len(N), 1)
        x_B = bq - (Aq @ x_N)
        c_B = np.array([c_[i] for i in B])
        v = c_B @ A_B_inv @ b

        c_N = np.array([c_[i] for i in sorted(list(N))])
        cq = c_N - c_B @ A_B_inv @ A_[:, N]

        z = v + cq @ x_N
        self.bq = bq
        self.cq = cq
        self.Aq = Aq

        if self.debug_print:
            print(" " * self.n_spaces * 2 + "z: ", z)
