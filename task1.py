import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def projection(u, w):
    return (np.vdot(u, w) / (np.linalg.norm(w)) ** 2) * w


def qr_decomposition(a):
    # Get dimension of matrix.
    n = len(a)
    # Create a copy of the matrix.
    cp = a.copy()

    # Initialize 2 matrices Q and R.
    Q = np.zeros(shape=(n, n))
    R = np.zeros(shape=(n, n))

    for i in range(n):
        u = a[:, i]
        u = u.astype('float32')
        w = u
        # Apply GS method.
        for k in range(i):
            u -= projection(w, Q[:, k])
        Q[:, i] = normalize(u)
        # Fill R at correct position.
        for j in range(i + 1):
            R[j, i] = np.vdot(Q[:, j], cp[:, i])

    # Return results.
    return Q, R


# Solves Ax=B
def solve(q, r, b):
    y = np.dot(q.T, b)
    n = len(r)
    x = np.zeros((4, 1))
    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(len(r[i]) - 1, i, -1):
            sum += r[i, j] * x[j]
        x[i] = (y[i] - sum) / r[i, i]

    return x


# Solves system using numpy
def check_with_numpy(a, b):
    q, r = np.linalg.qr(a)
    _y = np.dot(q.T, b)
    return np.linalg.solve(r, _y)


# Checks if all x satisfy all equations in the system by inserting them in each equation
def check_x(a, b, x):
    counter = 0
    for i in range(0, len(a)):
        _sum = 0
        for j in range(0, len(a[i])):
            _sum += a[i, j] * x[j]

        if abs(b[i] - _sum) > eps:
            counter += 1
        else:
            return False

    return True


# A = np.array([[1, 1, 1, 1],
#               [1, -1, -1, 1],
#               [2, 1, -1, 2],
#               [3, 1, 2, -1]])
#
# B = np.array([2, 0, 9, 7])

A = np.array([[0, 1, 2, 1],
              [6, -2, 3, 4],
              [0, 3, 4, -3],
              [0, -4, 3, 1]])

B = np.array([2, -15, 10, -2])

eps = 1e-10

Q, R = qr_decomposition(A.copy())
y = np.dot(Q.T, B)
temp = np.linalg.solve(R, y)
x = solve(Q, R, B.copy())
print(f"x = \n{x}")

x_correct = check_x(A, B, x)
print(f"x satisfy all equations in the system: {x_correct}\n")

x_check = check_with_numpy(A, B)
print(f"Solved with numpy: x = {x_check}")