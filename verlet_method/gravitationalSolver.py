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
from multiprocessing import Process, Queue
from threading import Thread
from verlet_cl import calc_verlet_opencl
dt = 1e0
gravi_const = 6.6742 * 1e-11 * (3600 ** 2)
global weights_mp
def _calc_accelerations_mp(x_pos, y_pos, amount_elements, i):
    v_equation_x, v_equation_y = 0, 0
    global weights_mp
    for j in range(amount_elements):
        if j == i:
            continue
        if x_pos[j] != x_pos[i]:
            v_equation_x +=  gravi_const * weights_mp[j] * (x_pos[j] - x_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
        if y_pos[j] != y_pos[i]:
            v_equation_y +=  gravi_const * weights_mp[j] * (y_pos[j] - y_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
    
    return v_equation_x, v_equation_y

def _calc_pos_prosess(N, amount_elements, init_position, pos_queue, speed_queue, return_pos_queue):
    '''
        Special function for multiprocessing. This necessery due to processes couldnt call same function from class
        Parametrs:
            N (int) - amount points in time
            amount_elements - amount elements
            init_speed (1d array) - initial speed (y coord) for elements
            pos_queue (Queue) - queue for pass and get position between processes
            speed_queue (Queue) - queue for pass and get speeds between processes
            return_pos_queue (Queue) - queue for collecting result positions in main process
    '''
    x_pos = np.zeros((N, amount_elements))
    y_pos = np.zeros((N, amount_elements))
    x_pos[0], y_pos[0] = init_position[:, 0], init_position[:, 1]
    for n in range(0, N - 1):
        acceleration_x, acceleration_y = np.zeros(amount_elements), np.zeros(amount_elements)
        for i in range(amount_elements):
            acceleration_x[i], acceleration_y[i] = _calc_accelerations_mp(x_pos[n], y_pos[n], amount_elements, i)
        curr_speed_x, curr_speed_y = speed_queue.get()
        for i in range(amount_elements):
            x_pos[n + 1, i] = x_pos[n, i] + curr_speed_x[i] * dt + acceleration_x[i] * (dt ** 2) / 2
            y_pos[n + 1, i] = y_pos[n, i] + curr_speed_y[i] * dt + acceleration_y[i] * (dt ** 2) / 2
        pos_queue.put([x_pos[n + 1], y_pos[n + 1], acceleration_x, acceleration_y])
    return_pos_queue.put([x_pos, y_pos])

def _clac_speed_prosess(N, amount_elements, init_speed, pos_queue, speed_queue, return_speed_queue):
    '''
        Special function for multiprocessing. This necessery due to processes couldnt call same function from class
        Parametrs:
            N (int) - amount points in time
            amount_elements - amount elements
            init_speed (1d array) - initial speed (y coord) for elements
            pos_queue (Queue) - queue for pass and get position between processes
            speed_queue (Queue) - queue for pass and get speeds between processes
            return_speed_queue (Queue) - queue for collecting result speed in main process
    '''
    speed_x = np.zeros((N, amount_elements))
    speed_y = np.zeros((N, amount_elements))
    speed_x[0], speed_y[0] = np.zeros(amount_elements), init_speed
    speed_queue.put([speed_x[0], speed_y[0]])
    for n in range(0, N - 1):
        curr_x_pos, curr_y_pos, acceleration_x, acceleration_y = pos_queue.get()
        for i in range(amount_elements):
            acceleration_x_next, acceleration_y_next = _calc_accelerations_mp(curr_x_pos, curr_y_pos, amount_elements, i)
            speed_x[n + 1, i] = speed_x[n, i]  + 0.5 * (acceleration_x_next + acceleration_x[i]) * dt
            speed_y[n + 1, i] = speed_y[n, i]  + 0.5 * (acceleration_y_next + acceleration_y[i]) * dt
        speed_queue.put([speed_x[n + 1], speed_y[n + 1]])
    return_speed_queue.put([speed_x, speed_y])


class GravitationalSolver():
    def __init__(self, weights, init_speed, init_position):
        
        self.amount_elements = len(weights)
        self.gravi_const = 6.6742 * 1e-11 * (3600 ** 2) #metrs^3 hours^-2 kilo^-1
        self.weights = weights
        global weights_mp
        weights_mp = weights #for multiprocessing
        self.init_speed = init_speed
        self.init_position = init_position

    def solve_odeint(self, max_time, dt):
        '''
            Solves the gravitational problem of N bodies by the odeint method
            
            Parametrs:
                max_time - max time in hours
                dt - delata in time
            
            Returns:
                sol - solution by odeint. It's array that contains x coord, y coord, speed for x coord, speed for y coord,
                      so shape is (max_time/dt, 4*amount_elemnets)

        '''
        t = np.linspace(0, max_time, int(max_time/dt))
        init_cond = np.concatenate((self.init_position[:, 0], self.init_position[:, 1], np.zeros(10), self.init_speed))
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
        '''
            Calculate accelerations for specific element according to formula.

            Parametrs:
                x_pos - array that contains x coordinates of all elements for current time
                y_pos - array that contains y coordinates of all elements for current time
                i - index of specific element
            
            Returns:
                v_equation_x - acceleration for x coordiante
                v_equation_y - acceleration for y coordiante
        '''
        
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
        '''     
            Implementation of the verlet method for the gravitational problem of N bodies.

            Parametrs:
                max_time - max time in hours
                dt - delata in time
            
            Returns:
                sol - solution by odeint. It's array that contains x coord, y coord, speed for x coord, speed for y coord,
                      so shape is (max_time/dt, 4*amount_elemnets)

        '''
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

    def solve_verlet_threading_pool(self, max_time, dt):
        '''
            Implementation of the verlet method using thread pool for the gravitational problem of N bodies.

            Parametrs:
                max_time - max time in hours
                dt - delata in time
            
            Returns:
                sol - solution by odeint. It's array that contains x coord, y coord, speed for x coord, speed for y coord,
                      so shape is (max_time/dt, 4*amount_elemnets)

        '''
        def _calc_positions_verlet(x_pos_curr, y_pos_curr, speed_x_curr, speed_y_curr, i, n):
            acceleration_x[i], acceleration_y[i] = self._calc_accelerations(x_pos_curr, y_pos_curr, i)
            x_pos[n + 1, i] = x_pos_curr[i] + speed_x_curr[i] * dt + acceleration_x[i] * (dt ** 2) / 2
            y_pos[n + 1, i] = y_pos_curr[i] + speed_y_curr[i] * dt + acceleration_y[i] * (dt ** 2) / 2

        def _calc_speeds_verlet(x_pos_curr, y_pos_curr, speed_x_curr, speed_y_curr, i, n):
            acceleration_x_next, acceleration_y_next = self._calc_accelerations(x_pos_curr, y_pos_curr, i)
            speed_x[n + 1, i] = speed_x_curr[i]  + 0.5 * (acceleration_x_next + acceleration_x[i]) * dt
            speed_y[n + 1, i] = speed_y_curr[i]  + 0.5 * (acceleration_y_next + acceleration_y[i]) * dt

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
                    jobs.append(executor.submit(_calc_positions_verlet, x_pos_curr=x_pos[n],
                                                                        y_pos_curr=y_pos[n], 
                                                                        speed_x_curr=speed_x[n], 
                                                                        speed_y_curr=speed_y[n], 
                                                                        i=i, 
                                                                        n=n))

            with ThreadPoolExecutor(max_workers=self.amount_elements) as executor:
                jobs = []
                for i in range(self.amount_elements):
                    jobs.append(executor.submit(_calc_speeds_verlet, x_pos_curr=x_pos[n + 1], 
                                                                     y_pos_curr=y_pos[n + 1], 
                                                                     speed_x_curr=speed_x[n], 
                                                                     speed_y_curr=speed_y[n], 
                                                                     i=i, 
                                                                     n=n))

        return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)
    
    def solve_verlet_threading(self, max_time, dt):
        '''
            Implementation of the verlet method with 2 threads for the gravitational problem of N bodies.
            First thread calc positions, second for speed. They are communicate using queues. (slight acceleration)

            Parametrs:
                max_time - max time in hours
                dt - delata in time
            
            Returns:
                sol - solution by odeint. It's array that contains x coord, y coord, speed for x coord, speed for y coord,
                      so shape is (max_time/dt, 4*amount_elemnets)

        '''
        N = int(max_time / dt)
        pos_queue, speed_queue = Queue(), Queue()
        return_pos_queue, return_speed_queue = Queue(), Queue()
        #create threads. First thread for calc position, second for speeds
        pos_thread = Thread(target=_calc_pos_prosess, args=(N, self.amount_elements, self.init_position, pos_queue, speed_queue, return_pos_queue))
        speed_thread = Thread(target=_clac_speed_prosess, args=(N, self.amount_elements, self.init_speed, pos_queue, speed_queue, return_speed_queue))

        pos_thread.start()
        speed_thread.start()

        #first gets answer due to problem with queue for a lot of elements
        x_pos, y_pos = return_pos_queue.get()
        pos_thread.join()

        speed_x, speed_y = return_speed_queue.get()
        speed_thread.join()

        return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)


    def solve_verlet_multitasking(self, max_time, dt):
        '''
            Implementation of the verlet method with 2 processes for the gravitational problem of N bodies.
            First process calc positions, second for speed. They are communicate using queues. (double acceleration)

            Parametrs:
                max_time - max time in hours
                dt - delata in time
            
            Returns:
                sol - solution by odeint. It's array that contains x coord, y coord, speed for x coord, speed for y coord,
                      so shape is (max_time/dt, 4*amount_elemnets)

        '''
        N = int(max_time / dt)
        pos_queue, speed_queue = Queue(), Queue()
        return_pos_queue, return_speed_queue = Queue(), Queue()
        #create processes. First process for calc position, second for speeds
        pos_prosess = Process(target=_calc_pos_prosess, args=(N, self.amount_elements, self.init_position, pos_queue, speed_queue, return_pos_queue))
        speed_prosess = Process(target=_clac_speed_prosess, args=(N, self.amount_elements, self.init_speed, pos_queue, speed_queue, return_speed_queue))

        pos_prosess.start()
        speed_prosess.start()

        #first gets answer due to problem with queue for a lot of elements
        x_pos, y_pos = return_pos_queue.get()
        pos_prosess.join()

        speed_x, speed_y = return_speed_queue.get()
        speed_prosess.join()
        return np.concatenate((x_pos, y_pos, speed_x, speed_y), axis=-1)
    
    def solve_verlet_cython(self, max_time, dt):
        '''
            Important!!! To use this function you need to run setup.py script.

            Implementation of the verlet method using Cython implementation for the gravitational problem of N bodies.

            Parametrs:
                max_time - max time in hours
                dt - delata in time
            
            Returns:
                sol - solution by odeint. It's array that contains x coord, y coord, speed for x coord, speed for y coord,
                      so shape is (max_time/dt, 4*amount_elemnets)

        '''
        return cython_solve.cypthon_solve(  max_time, 
                                            dt, 
                                            self.amount_elements, 
                                            self.init_position,
                                            self.init_speed,
                                            self.weights
                                            )
    
    def solve_verlet_cl(self, max_time, dt):
        '''

            Implementation of the verlet method using Opencl implementation for the gravitational problem of N bodies.

            Parametrs:
                max_time - max time in hours
                dt - delata in time
            
            Returns:
                sol - solution by odeint. It's array that contains x coord, y coord, speed for x coord, speed for y coord,
                      so shape is (max_time/dt, 4*amount_elemnets)

        '''
        return calc_verlet_opencl(  self.amount_elements, 
                                    max_time, 
                                    dt, 
                                    self.init_position,
                                    self.init_speed,
                                    self.weights
                                    )
