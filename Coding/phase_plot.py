import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from time import sleep


def duffing(coor, t, delta=0, m=1, k=1, beta=1, force=0, omega=1):
    x, y = coor
    dx = y
    dy = (-delta/m)*y - (k/m)*x - (beta/m)*(x**3) + (force/m)*np.cos(omega*t)
    return [dx, dy]


def plot_phase_diagram(func, x_vals, y_vals, plot_title, xlabel, ylabel, file_name, delta=0, m=1, k=1, beta=1, force=0, omega=1):
    plt.figure()
    plt.clf()
    plt.axis([x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]])
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    X, Y = np.meshgrid(x_vals, y_vals)

    t = 0

    u, v = np.zeros(X.shape), np.zeros(Y.shape)

    NI, NJ = X.shape

    for i in range(NI):
        for j in range(NJ):
            x = X[i, j]
            y = Y[i, j]
            yprime = func([x, y], t, delta, m, k, beta, force, omega)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]

    plt.quiver(X, Y, u, v, color='r')

    plt.xlim([x_vals[0], x_vals[-1]])
    plt.ylim([y_vals[0], y_vals[-1]])
    plt.savefig('images/' + file_name + '.png')

    for y20 in [-1.1, -0.9, -0.2, 0.2, 0.9, 1.1]:
        tspan = np.linspace(0, 25, 100)
        y0 = [y20, 0.0]
        ys = odeint(func, y0, tspan, args=(delta, m, k, beta, force, omega))
        plt.plot(ys[:, 0], ys[:, 1], 'b-')  # path
        plt.plot([ys[0, 0]], [ys[0, 1]], 'o')  # start
        plt.plot([ys[-1, 0]], [ys[-1, 1]], 's')  # end

    plt.savefig('images/' + file_name + '-lines.png')
    sleep(0.1)


if __name__ == "__main__":
    plot_title = "Vrije Duffing Oscillator"
    xlabel = "Distance"
    ylabel = "Velocity"

    ### DELTA == 0
    # NEGATIVE BETA
    x_vals = np.linspace(-1.5, -0.5, 20)
    y_vals = np.linspace(-0.5, 0.5, 20)

    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "ongedempt_beta_neg_0", beta=-1)

    x_vals = np.linspace(-0.5, 0.5, 20)
    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "ongedempt_beta_neg_1", beta=-1)

    x_vals = np.linspace(0.5, 1.5, 20)
    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "ongedempt_beta_neg_2", beta=-1)

    # POSITIVE BETA
    x_vals = np.linspace(-0.5, 0.5, 20)
    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "ongedempt_beta_pos")

    ### DELTA != 0
    # NEGATIVE BETA
    x_vals = np.linspace(-1.5, -0.5, 20)
    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "gedempt_beta_neg_0", delta=1, beta=-1)

    x_vals = np.linspace(-0.5, 0.5, 20)
    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "gedempt_beta_neg_1", delta=1, beta=-1)

    x_vals = np.linspace(0.5, 1.5, 20)
    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "gedempt_beta_neg_2", delta=1, beta=-1)

    # POSITIVE BETA
    x_vals = np.linspace(-0.5, 0.5, 20)
    plot_phase_diagram(duffing, x_vals, y_vals, plot_title, xlabel, ylabel, "gedempt_beta_pos", delta=1)
