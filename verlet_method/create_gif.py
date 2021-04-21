from gravitationalSolver import GravitationalSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from AnimatedScatter import AnimatedScatter
import time
import multiprocessing


if __name__ == '__main__':


    weights = np.array([1.989e30, 0.3281e24, 4.811e24, 5.9761e24, 0.6331e24, 1876.6431e24, 561.801e24, 86.0541e24, 101.5921e24, 0.01191e24]) #kilo
    speeds = np.array([0.0, 58.93, 35.18, 29.66, 23.6, 13.03, 10.16, 6.52, 5.43, 5.97]) * 3600 * 1000  #convert to metrs per hours
    positions =  np.array([[0.0, 0.0], [0.31, 0.1], [0.72, 0.11], [1.0, 0.12], [1.56, 0.13], [5.21, 0.14], [9.05, 0.15], [20.0, 0.16], [30.09, 0.17], [30.5, 0.18]]) * 1.496e11

    amount_repeat = 40
    #solver = GravitationalSolver(weights.repeat(amount_repeat), speeds.repeat(amount_repeat), positions.repeat(amount_repeat, axis=0))
    solver = GravitationalSolver(weights, speeds, positions)

    max_time = 10000


    start_time = time.time()
    sol = solver.solve_verlet(max_time=max_time, dt=1e0)
    print("--- %s seconds(pure) ---\n\n" % (time.time() - start_time))

    pos_x, pos_y = sol[:, :10], sol[:, 10:20] 

    amount_planets = 10
    colors = [[1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 1.0, 0.5],
            [0.8, 0.0, 0.3],
            [0.0, 0.3, 0.6]
    ]
    scale = np.array([2., 0.3, 0.35, 0.45, 0.6, 0.75, 0.9, 1.1, 1.3, 0.2]) * 10
    anim = AnimatedScatter(np.array([pos_x[::100]/ 1.496e11, pos_y[::100]/ 1.496e11]), scale=scale, colors=colors).ani
    anim.save('solar_sytem_verlet.gif', writer='imagemagick')