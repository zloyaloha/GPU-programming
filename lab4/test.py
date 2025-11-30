import numpy as np
import random

def generate_well_conditioned_matrix(n, low=-10, high=10, seed=None):
    """
    Генерирует случайную невырожденную матрицу n x n с "адекватным" детерминантом.
    """
    if seed is not None:
        np.random.seed(seed)

    # Генерируем случайную матрицу
    A = np.random.uniform(low, high, size=(n, n))

    # Для улучшения обусловленности можно сделать диагонально доминирующую
    for i in range(n):
        A[i, i] += n * (high - low) / 2

    return A

def save_matrix_to_file(matrix, filename):
    """
    Сохраняет матрицу в файл в формате:
    n
    a11 a12 ... a1n
    ...
    an1 an2 ... ann
    """
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
