import numpy as np
cimport numpy as np

cdef double gravi_const = 6.6742 * 1e-11 * (3600 ** 2)
def _calc_accelerations(np.ndarray x_pos, np.ndarray y_pos, int amount_elements, np.ndarray weights, int i):
        
    v_equation_x, v_equation_y = 0, 0
    for j in range(amount_elements):
        if j == i:
            continue
        if x_pos[j] != x_pos[i]:
            v_equation_x +=  gravi_const * weights[j] * (x_pos[j] - x_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
        if y_pos[j] != y_pos[i]:
            v_equation_y +=  gravi_const * weights[j] * (y_pos[j] - y_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
    
    return v_equation_x, v_equation_y

def cypthon_solve(int max_time, float dt, int amount_elements, np.ndarray init_position, np.ndarray init_speed, np.ndarray weights):
    N = int(max_time / dt)
    x_pos = np.zeros((N, amount_elements))
    y_pos = np.zeros((N, amount_elements))
    speed_x = np.zeros((N, amount_elements))
    speed_y = np.zeros((N, amount_elements))
    x_pos[0] = init_position[:, 0]
    y_pos[0] = init_position[:, 1]
    speed_x[0] = np.zeros(amount_elements)
    speed_y[0] = init_speed
    for n in range(0, N - 1):
        acceleration_x, acceleration_y = np.zeros(amount_elements), np.zeros(amount_elements)
        for i in range(amount_elements):
            acceleration_x[i], acceleration_y[i] = _calc_accelerations(x_pos[n], y_pos[n], amount_elements, weights, i)
            x_pos[n + 1, i] = x_pos[n, i] + speed_x[n, i] * dt + acceleration_x[i] * (dt ** 2) / 2
            y_pos[n + 1, i] = y_pos[n, i] + speed_y[n, i] * dt + acceleration_y[i] * (dt ** 2) / 2

        for i in range(amount_elements):
            acceleration_x_next, acceleration_y_next = _calc_accelerations(x_pos[n + 1], y_pos[n + 1], amount_elements, weights, i)
            speed_x[n + 1, i] = speed_x[n, i]  + 0.5 * (acceleration_x_next + acceleration_x[i]) * dt
            speed_y[n + 1, i] = speed_y[n, i]  + 0.5 * (acceleration_y_next + acceleration_y[i]) * dt

    return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)
  