import numpy as np
cimport numpy as np


def cypthon_solve(int max_time, float dt, int amount_elements, np.ndarray init_position):
    N = int(max_time / dt)
    x_pos = np.zeros((N, amount_elements))
    y_pos = np.zeros((N, amount_elements))
    speed_x = np.zeros((N, amount_elements))
    speed_y = np.zeros((N, amount_elements))
    return 0    
  