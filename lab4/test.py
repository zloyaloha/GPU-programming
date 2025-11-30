import numpy as np
import random

def generate_well_conditioned_matrix(n, low=-10, high=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A = np.random.uniform(low, high, size=(n, n))

    for i in range(n):
        A[i, i] += n * (high - low) / 2

    return A

def save_matrix_to_file(matrix, filename):
    n = matrix.shape[0]
    with open(filename, "w") as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")

# Пример использования
n = 3000
matrix = generate_well_conditioned_matrix(n, low=1, high=10, seed=random.randint(1, 10000))
save_matrix_to_file(matrix, "/home/zloyaloha/programming/Numerical-analysis/build/matrix.txt")

print("Матрица сгенерирована и сохранена в 'matrix.txt':")
print("det(A) =", np.linalg.det(matrix))
