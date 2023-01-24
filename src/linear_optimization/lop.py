import numpy as np


class SLACK:

    def __init__(self, B, Aq, bq, cq, v, name):
        self.B = B
        self.Aq = Aq
        self.bq = bq
        self.cq = cq
        self.v = v
        self.name = name

    def info(self):
        s = "Slackform: %s \nB:\n %s \nAq: \n %s \nbq:\n %s \ncq:\n %s \n %s \n"
        return s % (self.name, self.B, self.Aq, self.bq, self.cq, self.v)

    def __str__(self):
        return self.info()

    def __repr__(self):
        return self.__str__()


class LOP:

    def __init__(self, A, b, c, name):
        for arg in [A, b, c]:
            assert type(arg).__module__ == 'numpy'

        assert A.shape[0] == b.shape[0]
        assert A.shape[1] == c.shape[0]
        assert b.shape[0] == A.shape[0]
        assert b.shape[1] == 1

        self.A = A
        self.b = b
        self.c = c
        self.name = name

        self.m = self.b.shape[0]
        self.n = self.c.shape[0]

        # print(self.info())

    def info(self):
        s = "LOP: %s \n A: \n %s \n b:\n %s \n c:\n %s \n"
        return s % (self.name, self.A, self.b, self.c)

    def __str__(self):
        return self.info()

    def __repr__(self):
        return self.__str__()


L1 = LOP(A=np.array([3, 2, 1, 2, 1, 1, 1, 1, 4, 3, 3, 4]).reshape(3, 4),
         b=np.array([225, 117, 420]).reshape(3, 1),
         c=np.array([19, 13, 12, 17]),
         name="Möbelfabrick Instanz")

L2 = LOP(A=np.array([
    [17., -1., -6., -14., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 22., 4., -1., -9., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 27., 9., 4., -4.],
    [-1., -1., -1., -1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
    [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
    [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]]),
    b=np.array([
        [0],
        [0],
        [0],
        [-15000],
        [10000],
        [4000],
        [5050],
        [7100],
        [4300]]),
    c=np.array([9.97, 7.84, 4.64, 4.24, 11.93, 9.8,
                6.6, 6.2, 14.13, 12., 8.8, 8.4]),
    name="Mischungsproblem Ölraffinerie")

L3 = LOP(A=np.array([[2, 3, 1],
                     [4, 1, 2],
                     [3, 4, 2]]),
         b=np.array([5, 11, 8]).reshape(3, 1),
         c=np.array([5, 4, 3]),
         name="Aufgabe 3.1")
