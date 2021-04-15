import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import multiprocessing
from functools import partial
import cython_solve
dt = 1e0
gravi_const = 6.6742 * 1e-11 * (3600 ** 2)
def _calc_accelerations_mp(x_pos, y_pos, weights, i):
    v_equation_x, v_equation_y = 0, 0
    for j in range(len(x_pos)):
        if j == i:
            continue
        if x_pos[j] != x_pos[i]:
            v_equation_x +=  gravi_const * weights[j] * (x_pos[j] - x_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
        if y_pos[j] != y_pos[i]:
            v_equation_y +=  gravi_const * weights[j] * (y_pos[j] - y_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
    return v_equation_x, v_equation_y

def _calc_speeds_verlet_mp(x_pos, y_pos, speed_x, speed_y, acceleration_x, acceleration_y, weights, i):
    acceleration_x_next, acceleration_y_next = _calc_accelerations_mp(x_pos, y_pos, weights, i)
    tmp1 = speed_x[i]  + 0.5 * (acceleration_x_next + acceleration_x[i]) * dt
    tmp2 = speed_y[i]  + 0.5 * (acceleration_y_next + acceleration_y[i]) * dt
    return tmp1, tmp2

def _calc_positions_verlet_mp(x_pos, y_pos, speed_x, speed_y, weights, i):
    acceleration_x, acceleration_y = _calc_accelerations_mp(x_pos, y_pos, weights, i)
    tmp1 = x_pos[i] + speed_x[i] * dt + acceleration_x * (dt ** 2) / 2
    tmp2 = y_pos[i] + speed_y[i] * dt + acceleration_y * (dt ** 2) / 2
    return tmp1, tmp2, acceleration_x, acceleration_y

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
            if x_pos[j] != x_pos[i]:
                v_equation_x +=  self.gravi_const * self.weights[j] * (x_pos[j] - x_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
            if y_pos[j] != y_pos[i]:
                v_equation_y +=  self.gravi_const * self.weights[j] * (y_pos[j] - y_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
       
        return v_equation_x, v_equation_y

    def solve_verlet(self, max_time, dt):
        N = int(max_time / dt)
        x_pos = np.zeros((N, self.amount_elements))
        y_pos = np.zeros((N, self.amount_elements))
        speed_x = np.zeros((N, self.amount_elements))
        speed_y = np.zeros((N, self.amount_elements))

        x_pos[0], y_pos[0] = self.init_position[:, 0], self.init_position[:, 1]
        speed_x[0], speed_y[0] =  np.zeros(self.amount_elements), self.init_speed
        for n in range(0, N - 1):
            acceleration_x, acceleration_y = np.zeros(self.amount_elements), np.zeros(self.amount_elements)
            for i in range(self.amount_elements):
                acceleration_x[i], acceleration_y[i] = self._calc_accelerations(x_pos[n], y_pos[n], i)
                x_pos[n + 1, i] = x_pos[n, i] + speed_x[n, i] * dt + acceleration_x[i] * (dt ** 2) / 2
                y_pos[n + 1, i] = y_pos[n, i] + speed_y[n, i] * dt + acceleration_y[i] * (dt ** 2) / 2

            for i in range(self.amount_elements):
                acceleration_x_next, acceleration_y_next = self._calc_accelerations(x_pos[n + 1], y_pos[n + 1], i)
                speed_x[n + 1, i] = speed_x[n, i]  + 0.5 * (acceleration_x_next + acceleration_x[i]) * dt
                speed_y[n + 1, i] = speed_y[n, i]  + 0.5 * (acceleration_y_next + acceleration_y[i]) * dt

        return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)



    def solve_verlet_threading(self, max_time, dt):
        def _calc_positions_verlet(x_pos, y_pos, speed_x, speed_y, i):
            acceleration_x, acceleration_y = self._calc_accelerations(x_pos, y_pos, i)
            tmp1 = x_pos[i] + speed_x[i] * dt + acceleration_x * (dt ** 2) / 2
            tmp2 = y_pos[i] + speed_y[i] * dt + acceleration_y * (dt ** 2) / 2
            return tmp1, tmp2, acceleration_x, acceleration_y, i

        def _calc_speeds_verlet(x_pos, y_pos, speed_x, speed_y, i):
            acceleration_x_next, acceleration_y_next = self._calc_accelerations(x_pos, y_pos, i)
            tmp1 = speed_x[i]  + 0.5 * (acceleration_x_next + acceleration_x[i]) * dt
            tmp2 = speed_y[i]  + 0.5 * (acceleration_y_next + acceleration_y[i]) * dt
            return tmp1, tmp2, i

        N = int(max_time / dt)
        x_pos = np.zeros((N, self.amount_elements))
        y_pos = np.zeros((N, self.amount_elements))
        speed_x = np.zeros((N, self.amount_elements))
        speed_y = np.zeros((N, self.amount_elements))


        x_pos[0], y_pos[0] = self.init_position[:, 0], self.init_position[:, 1]
        speed_x[0], speed_y[0] =  np.zeros(self.amount_elements), self.init_speed
        
        for n in range(N - 1):
            acceleration_x, acceleration_y = np.zeros(self.amount_elements), np.zeros(self.amount_elements)
            with ThreadPoolExecutor(max_workers=self.amount_elements) as executor:
                jobs = []
                for i in range(self.amount_elements):
                    jobs.append(executor.submit(_calc_positions_verlet, x_pos=x_pos[n], y_pos=y_pos[n], speed_x=speed_x[n], speed_y=speed_y[n], i=i))

                for job in futures.as_completed(jobs):
                    result_done = job.result()
                    i = result_done[-1]
                    x_pos[n + 1, i], y_pos[n + 1, i] = result_done[0], result_done[1]
                    acceleration_x[i], acceleration_y[i] = result_done[2], result_done[3]

            with ThreadPoolExecutor(max_workers=self.amount_elements) as executor:
                jobs = []
                for i in range(self.amount_elements):
                    jobs.append(executor.submit(_calc_speeds_verlet, x_pos=x_pos[n + 1], y_pos=y_pos[n + 1], speed_x=speed_x[n], speed_y=speed_y[n], i=i))

                for job in futures.as_completed(jobs):
                    result_done = job.result()
                    i = result_done[-1]
                    speed_x[n + 1, i], speed_y[n + 1, i] = result_done[0], result_done[1]
       
        return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)
    
    def solve_verlet_multitasking(self, max_time, dt):
        N = int(max_time / dt)
        x_pos = np.zeros((N, self.amount_elements))
        y_pos = np.zeros((N, self.amount_elements))
        speed_x = np.zeros((N, self.amount_elements))
        speed_y = np.zeros((N, self.amount_elements))

        x_pos[0], y_pos[0] = self.init_position[:, 0], self.init_position[:, 1]
        speed_x[0], speed_y[0] =  np.zeros(self.amount_elements), self.init_speed
        tmp_lst1 = [(np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), i) for i in range(self.amount_elements)]
        tmp_lst2 = [(np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), np.zeros(self.amount_elements), i) for i in range(self.amount_elements)]
        for n in range(0, N - 1):
           # acceleration_x, acceleration_y = np.zeros(self.amount_elements), np.zeros(self.amount_elements)

            #tmp_lst = [(x_pos[n], y_pos[n], speed_x[n], speed_y[n], self.weights, i) for i in range(self.amount_elements)]
            
            
            with multiprocessing.Pool(processes=self.amount_elements) as pool:
                results = np.array(pool.starmap(_calc_positions_verlet_mp, tmp_lst1))
                x_pos[n + 1], y_pos[n + 1] = results[:, 0], results[:, 1]
                acceleration_x, acceleration_y  = results[:, 2], results[:, 3] 

            #tmp_lst = [(x_pos[n + 1], y_pos[n + 1], speed_x[n], speed_y[n], acceleration_x, acceleration_y, self.weights, i) for i in range(self.amount_elements)]
            
            with multiprocessing.Pool(processes=self.amount_elements) as pool:
                results = np.array(pool.starmap(_calc_speeds_verlet_mp, tmp_lst2))
                speed_x[n + 1], speed_y[n + 1] = results[:, 0], results[:, 1]
        return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)
    
    def solve_verlet_cython(self, max_time, dt):
        return cython_solve.cypthon_solve(max_time, 
                                            dt, 
                                            self.amount_elements, 
                                            self.init_position,
                                            self.init_speed,
                                            self.weights
                                            )

    def plot_solution(self, solution, i):
        solution = solution[i]
        positions_x, positions_y = solution[:self.amount_elements], solution[self.amount_elements:2*self.amount_elements]
        plt.scatter(positions_x / 1.496e11, positions_y / 1.496e11)
        plt.show()