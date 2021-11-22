import numpy as np


def duffing(coor, t, delta=0, m=1, k=1, beta=1):
    x, y = coor
    dx = y
    dy = -(k/m)*x - (beta/m)*(x**3)
    return [dx, dy]


def euler(func, start_val, step_size, start=0, end=50):
    # delta=0, m=1, k=4, beta=-0.04 == -1/25
    duffing_vars = {"delta": 0, "m": 1, "k": 4, "beta": -0.04}
    # t = 50
    x, y = start_val

    t = start
    while t < end:
        dx, dy = func((x, y), t, **duffing_vars)
        x = x + step_size * dx
        y = y + step_size * dy

        t += step_size

    return x, y


def euler_verbeterd(func, start_val, step_size, start=0, end=50):
    # delta=0, m=1, k=4, beta=-0.04 == -1/25
    duffing_vars = {"delta": 0, "m": 1, "k": 4, "beta": -0.04}
    # t = 50
    x, y = start_val

    t = start
    while t < end:
        x_k1, y_k1 = func((x, y), t, **duffing_vars)
        x_k2, y_k2 = func((x + step_size*x_k1, y + step_size*y_k1), t + step_size, **duffing_vars)

        x = x + step_size * (x_k1 + x_k2)/2
        y = y + step_size * (y_k1 + y_k2)/2

        t += step_size

    return x, y


def runge_kutta_4(func, start_val, step_size, start=0, end=50):
    # delta=0, m=1, k=4, beta=-0.04 == -1/25
    duffing_vars = {"delta": 0, "m": 1, "k": 4, "beta": -0.04}
    # t = 50
    x, y = start_val

    t = start
    while t < end:
        x_k1, y_k1 = func((x, y), t, **duffing_vars)
        x_k2, y_k2 = func((x + step_size * x_k1 / 2, y + step_size * y_k1 / 2), t + step_size/2, **duffing_vars)
        x_k3, y_k3 = func((x + step_size * x_k2 / 2, y + step_size * y_k2 / 2), t + step_size / 2, **duffing_vars)
        x_k4, y_k4 = func((x + step_size * x_k3, y + step_size * y_k3), t + step_size)

        x = x + (step_size/6) * (x_k1 + 2*x_k2 + 2*x_k3 + x_k4) / 2
        y = y + (step_size/6) * (y_k1 + 2*y_k2 + 2*y_k3 + y_k4) / 2
        t += step_size

    return x, y


def iterate(iterator, func, start_val, start_step=0.02, start=1, end=50):
    x0, y0 = iterator(func, start_val, start_step, start, end)
    x1, y1 = iterator(func, start_val, start_step/2, start, end)

    step = start_step / 2
    division_counter = 1
    while abs(x0 - x1) >= 0.001 or abs(y0 - y1) >= 0.001:
        x0 = x1
        y0 = y1
        step = step/2
        division_counter += 1
        x1, y1 = iterator(func, start_val, step, start, end)

    print("insignificant difference at", division_counter, "divisions of", start_step)
    return x1, y1


if __name__ == "__main__":
    start_vals = [(1, 0), (8, 0), (-6, 2)]

    for start_val in start_vals:
        print("START VALUE:", start_val)
        print(iterate(euler, duffing, start_val))
        print(iterate(euler_verbeterd, duffing, start_val))
        print(iterate(runge_kutta_4, duffing, start_val))
