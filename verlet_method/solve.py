from gravitationalSolver import GravitationalSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from AnimatedScatter import AnimatedScatter

weights = np.array([1.989e30, 0.3281e24, 4.811e24, 5.9761e24, 0.6331e24, 1876.6431e24, 561.801e24, 86.0541e24, 101.5921e24, 0.01191e24]) #kilo
speeds = np.array([0.0, 58.93, 35.18, 29.66, 23.6, 13.03, 10.16, 6.52, 5.43, 5.97]) * 3600 * 1000  #convert to metrs per hours
positions =  np.array([[0.0, 0.0], [0.31, 0.1], [0.72, 0.11], [1.0, 0.12], [1.56, 0.13], [5.21, 0.14], [9.05, 0.15], [20.0, 0.16], [30.09, 0.17], [30.5, 0.18]]) * 1.496e11

solver = GravitationalSolver(weights, speeds, positions)

#sol = solver.solve_verlet(max_time=10000, dt=1e0)
sol = solver.solve_odeint()
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

anim = AnimatedScatter(np.array([pos_x[::10]/ 1.496e11, pos_y[::10]/ 1.496e11]), colors=colors)
plt.show()