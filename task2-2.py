import numpy as np

np.warnings.filterwarnings('ignore')


def newton(x):
    for i in range(max_iterations):
        delta_x, a, b, c = np.linalg.lstsq(-df(x), f(x))
        x = x + np.array(delta_x).reshape(1, 4)
        x = np.array([x[0, 0], x[0, 1], x[0, 2], x[0, 3]])

        precision = np.linalg.norm(delta_x) / (np.linalg.norm(x) + np.linalg.norm(delta_x))
        print(f"Iteration: {i} Precision: {precision}")

        if precision < eps:
            print(f"Solution: {x}")
            print(f"f(x) = {f(x)}")
            return x
        elif i == max_iterations:
            print(f"Set precision not reached. Last x = {x}")
            return


# def f(x):
#     return np.asmatrix([
#         [x[0] + 2 * x[1] + x[2] + 4 * x[3] - 20.7],
#         [x[0] ** 2 + 2 * x[0] * x[1] + x[3] ** 3 - 15.88],
#         [x[0] ** 3 + x[2] ** 2 + x[3] - 21.218],
#         [3 * x[1] + x[2] * x[3] - 7.9]
#     ])
#
#
# def df(x):
#     return np.asmatrix([
#         [1, 2, 1, 4],
#         [2 * x[0] + 2 * x[1], 2 * x[0], 0, 3 * x[3] ** 2],
#         [3 * x[0] ** 2, 0, 2 * x[2], 1],
#         [0, 3, x[3], x[2]]
#     ])


def f(x):
    return np.asmatrix([
        [5 * x[0] + x[1] + x[2] + 4 * x[3] + 5],
        [-x[0] ** 2 + x[2] ** 2 + 5],
        [4 * x[2] ** 3 - x[3] ** 2 - 3 * x[1] * x[3] - 28],
        [x[0] - 3 * x[1] + 4 * x[2] - x[3] - 3]
    ])


def df(x):
    return np.asmatrix([
        [5, 1, 1, 4],
        [-2 * x[0], 0, 2 * x[2], 0],
        [0, -3 * x[3], 12 * (x[2] ** 2), -2 * x[3] - 3 * x[1]],
        [1, -3, 4, -1]
    ])


initial_guess = np.array([1, 1, 1, 1])
max_iterations = 200
eps = 1e-10

newton(initial_guess)