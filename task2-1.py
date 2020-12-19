from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

np.warnings.filterwarnings('ignore')


# https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contour.html

# Shows z1 function graph
def show_z1_graph():
    fig = plt.figure()
    fig.canvas.set_window_title('Z1')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(XX, YY, Z1, cmap=cm.coolwarm,
                           alpha=0.5)
    surfZ = ax.plot_surface(XX, YY, np.zeros(np.shape(Z1)), antialiased=False, alpha=0.2)
    cp = ax.contour(X, Y, Z1, levels=0, colors='red')
    plt.show()


# Shows z2 function graph
def show_z2_graph():
    fig = plt.figure()
    fig.canvas.set_window_title('Z2')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(XX, YY, Z2, cmap=cm.summer,
                           antialiased=False, alpha=0.5)
    surf_z = ax.plot_surface(XX, YY, np.zeros(np.shape(Z1)), antialiased=False, alpha=0.2)
    cp = ax.contour(X, Y, Z2, levels=0, colors='green')
    plt.show()


# Solves system graphically
def show_f_roots():
    fig = plt.figure()
    fig.canvas.set_window_title('Result')
    ax = fig.gca()
    ax.grid(color='#C0C0C0', linestyle='-', linewidth=0.5)
    cp = ax.contour(X, Y, Z1, levels=0, colors='red')
    cp = ax.contour(X, Y, Z2, levels=0, colors='green')
    plt.show()


# Solves system using newton's method
def newton(x):
    ff = f(x)
    dff = df(x)
    for i in range(max_iterations):
        dff = df(x)
        delta_x, a, b, c = np.linalg.lstsq(-dff, ff)

        x1 = x.reshape(2, 1) + alpha * delta_x
        ff1 = f([x1[0, 0], x1[1, 0]])

        precision = np.linalg.norm(delta_x) / (np.linalg.norm(x) + np.linalg.norm(delta_x))
        print(f"Iteration: {i} Precision: {precision}")

        if precision < eps:
            print(f"Solution: {x}")
            return x
        elif i == max_iterations:
            print(f"Set precision not reached. Last x = {x}")
            return

        x = np.array([x1[0, 0], x1[1, 0]])
        ff = ff1


# System of nonlinear equations
# def f(x):
#     return np.asmatrix([
#         [x[0] ** 2 + x[1] ** 2 - 2],
#         [x[0] ** 2 - x[1] ** 2]
#     ])


def f(x):
    return np.asmatrix([
        [np.e ** -((((x[0] + 2) ** 2) + 2 * x[1] ** 2) / 4) -0.1],
        [x[0] ** 2 *  x[1] ** 2 + x[0] -8]
    ])


def df(x):
    return np.asmatrix([
        [-((x[0] + 2) / 2 + x[1]) * np.e ** -((((x[0] + 2) ** 2) + 2 * x[1] ** 2) / 4)],
        [2 * x[0] * 2 * x[1] + 1]
    ])





# Used for showing graphs
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
XX, YY = np.meshgrid(X, Y)

Z1 = XX ** 2 + 10 * (np.sin(XX) + np.cos(YY)) ** 2 - 10
Z2 = (YY - 3) ** 2 + XX - 8

show_z1_graph()
show_z2_graph()

show_f_roots()

alpha = 1
max_iterations = 200
eps = 1e-10
initial_x = np.array([1, 1])  # Initial guess

# Solves system and checks with f function
result = newton(initial_x)
print(f(result))