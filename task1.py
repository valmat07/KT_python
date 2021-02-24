import numpy as np
import matplotlib.pyplot as plt
import time


def create_matrix(dim):

    eps = 1e-5
    tmp_matrix = np.random.rand(dim, dim)

    while np.linalg.det(tmp_matrix) < eps:
        tmp_matrix = np.random.rand(dim, dim)

    return tmp_matrix

def gaus_solve_system(matrix, b, amount_repeats = 3):

    solver_time = 0    
    
    system_eq = np.concatenate((matrix, b.T[:, np.newaxis]), axis=1)
    matrix_dim = matrix.shape[0]
    for i in range(amount_repeats):
        start_time = time.time()
        #forward pass
        for j in range(matrix_dim):
            for i in range(j, matrix_dim - 1):
                system_eq[i + 1] = system_eq[i + 1] - system_eq[i + 1, j] * system_eq[j]/system_eq[j, j]

        #backward pass
        x = np.zeros(matrix_dim)
        for i in range(matrix_dim - 1, -1, -1):
            x[i] = system_eq[i, matrix_dim] / system_eq[i, i]
            for k in range(i - 1, -1, -1):
                system_eq[k, matrix_dim] -= system_eq[k, i] * x[i]
        
        solver_time += time.time() - start_time

    return solver_time / amount_repeats
def plot_results(dim_array, time_array):

    plt.title('Dependence of computation time on dimension')
    plt.plot(dim_array, time_array)
    plt.show()


if __name__ == "__main__":
    dims = [10 ** i for i in range(1, 4)]
    times = np.zeros(len(dims))
    for i, dim in enumerate(dims):
        matrix = create_matrix(dim)
        b = np.random.rand(dim)
        times[i] = gaus_solve_system(matrix, b)
    plot_results(dims, times)