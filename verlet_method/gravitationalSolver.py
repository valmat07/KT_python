import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
class GravitationalSolver():
    def __init__(self, weights, init_speed, init_position):
        self.amount_elements = len(weights)
        self.gravi_const = 6.6742 * 1e-11 * (3600 ** 2) #metrs^3 hours^-2 kilo^-1
        self.weights = weights
        self.init_speed = init_speed
        self.init_position = init_position

    def solve_odeint(self):
        t = np.linspace(0, 1000, 1000)
        init_cond = np.concatenate((self.init_position[:, 0], self.init_position[:, 1], np.zeros(10), self.init_speed))
        #solution_class = solve_ivp(self._ode_equation, t, init_cond, dense_output=True)
        #sol = solution_class.sol(t).T
        sol = odeint(self._ode_equation, init_cond, t)
        return sol
        
    def _ode_equation(self, y, t):
        drdt_x, dvdt_x, dvdt_y, drdt_y  = [], [], [], []
        positions_x, positions_y = y[:self.amount_elements], y[self.amount_elements:2*self.amount_elements]   
        speeds_x, speeds_y = y[2*self.amount_elements:3*self.amount_elements], y[3*self.amount_elements:] 

        for i in range(self.amount_elements):
            drdt_x.append(speeds_x[i])
            drdt_y.append(speeds_y[i])
            
            acceleration_x, acceleration_y = self._calc_accelerations(positions_x, positions_y, i)
            dvdt_x.append(acceleration_x)
            dvdt_y.append(acceleration_y)

        dydt = drdt_x
        dydt.extend(drdt_y)
        dydt.extend(dvdt_x)
        dydt.extend(dvdt_y)
        return dydt

    def _calc_accelerations(self, x_pos, y_pos, i):
        v_equation_x, v_equation_y = 0, 0

        for j in range(self.amount_elements):
            if j == i:
                continue
            v_equation_x +=  self.gravi_const * self.weights[j] * (x_pos[j] - x_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
            v_equation_y +=  self.gravi_const * self.weights[j] * (y_pos[j] - y_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
        return v_equation_x, v_equation_y

    def solve_verlet(self, max_time, dt):
        N = int(max_time / dt)
        x_pos = np.zeros((N, self.amount_elements))
        y_pos = np.zeros((N, self.amount_elements))
        speed_x = np.zeros((N, self.amount_elements))
        speed_y = np.zeros((N, self.amount_elements))

        x_pos[0], y_pos[0] = self.init_position[:, 0], self.init_position[:, 1]
        speed_x[0], speed_y[0] =  np.zeros(10), self.init_speed
        for n in range(0, N - 1):
            for i in range(self.amount_elements):
                acceleration_x, acceleration_y = self._calc_accelerations(x_pos[n], y_pos[n], i)
                x_pos[n + 1, i] = x_pos[n, i] + speed_x[n, i] * dt + acceleration_x * (dt ** 2) / 2
                y_pos[n + 1, i] = y_pos[n, i] + speed_y[n, i] * dt + acceleration_y * (dt ** 2) / 2

            for i in range(self.amount_elements):
                acceleration_x, acceleration_y = self._calc_accelerations(x_pos[n], y_pos[n], i)
                acceleration_x_next, acceleration_y_next = self._calc_accelerations(x_pos[n + 1], y_pos[n + 1], i)
                speed_x[n + 1, i] = speed_x[n, i]  + 0.5 * (acceleration_x_next + acceleration_x) * dt
                speed_y[n + 1, i] = speed_y[n, i]  + 0.5 * (acceleration_y_next + acceleration_y) * dt

        return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)
    def solve_verlet_threading(self):
        pass

    def solve_verlet_multitasking(self):
        pass
    
    def plot_solution(self, solution, i):
        solution = solution[i]
        positions_x, positions_y = solution[:self.amount_elements], solution[self.amount_elements:2*self.amount_elements]
        plt.scatter(positions_x / 1.496e11, positions_y / 1.496e11)
        plt.show()