from numpy import mean
from numpy.linalg import eig # eigenvalues_and_eigenvectors

from src.algebra import matrix_transpose, matrix_multiplication
from src.statistic import covariance_matrix

T = matrix_transpose


def principal_component_analysis(A):
    column_means = [mean(column) for column in T(A)]

    # center columns by subtracting column mean
    centers = [[row[i] - column_means[i]
                for i in range(len(A[0]))] for row in A]

    # calculate covariance matrix of centered matrix
    cov = covariance_matrix(centers)

    # calculate eigendecomposition of covariance matrix
    values, vectors = eig(cov)

    # project data into subspace via matrix multiplication
    return T(matrix_multiplication(T(vectors), T(centers)))


if __name__ == '__main__':

    A = [[1, 2],
         [3, 4],
         [5, 6]]

    print(principal_component_analysis(A))
