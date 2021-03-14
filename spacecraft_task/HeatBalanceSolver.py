import pandas as pd
from scipy.integrate import odeint
import numpy as np
from scipy.optimize import fsolve

class HeatBalanceSolver():
    def __init__(self, parametrs_file_name, surfaces_area, area_btw_surfaces):
        super(HeatBalanceSolver, self).__init__()
        self.parametrs_file_name = parametrs_file_name
        self.surfaces_area = surfaces_area
        self.area_btw_surfaces = area_btw_surfaces
        self.c_0 = 5.67
        self.amount_elemnts = len(self.area_btw_surfaces)

        self.parse_parametrs()

    def _equation(self, T, t):
        dTdt = []
        for i in range(self.amount_elemnts):
            tmp_eq = 0
            for j in range(self.amount_elemnts):
                k = self.lambdas[i, j] * self.area_btw_surfaces[i, j]
                tmp_eq += k * (T[j] - T[i])
            q_e = -self.epsilon[i] * self.surfaces_area[i] * self.c_0 * ((T[i] / 100) ** 4)

            tmp_eq += q_e + self.A[i]*(20 + 3 * np.cos(t/4))

            dTdt.append(tmp_eq / self.c[i])
        return dTdt

    def parse_parametrs(self):
        parametrs_df = pd.read_csv(self.parametrs_file_name)
        self.epsilon = np.array(parametrs_df['eps'])
        self.c = np.array(parametrs_df['c'])

        self.q_r = []
        self.lambdas = np.zeros((len(parametrs_df['lambdas_1_x']), len(parametrs_df['lambdas_1_x'])))
        
        for i in range(self.lambdas.shape[0] - 1):
            self.lambdas[i] = np.array(parametrs_df['lambdas_{}_x'.format(i + 1)])
        
        for i in range(1, self.amount_elemnts):
            for j in range(i, self.amount_elemnts):
                self.lambdas[j, i] = self.lambdas[i, j]

        self.A = parametrs_df['A']
    def _stationary_equation(self, T):
        equations = []
        for i in range(self.amount_elemnts):
            tmp_eq = 0
            for j in range(self.amount_elemnts):
                k = self.lambdas[i, j] * self.area_btw_surfaces[i, j]
                tmp_eq += k * (T[j] - T[i])
            q_e = -self.epsilon[i] * self.surfaces_area[i] * self.c_0 * ((T[i] / 100) ** 4)
            tmp_eq += q_e
            equations.append(tmp_eq / self.c[i])
        return equations

    def get_stationary_solution(self):
        return fsolve(self._stationary_equation, np.zeros(self.amount_elemnts))
    
    def save_solution(self, solution, time):
        solution_dict = {}
        solution_dict['time'] = time.tolist()
        for i in range(solution.shape[1]):
            solution_dict['temp_elemnt_{}'.format(i + 1)] = solution[:, i]
        df = pd.DataFrame(solution_dict)
        df.to_csv('solution.csv')




    def solve(self, t0, t1):
        N = 100
        t = np.linspace(t0, t1, N)
        init_cond = self.get_stationary_solution()
        sol = odeint(self._equation, init_cond, t)
        self.save_solution(sol, t)
        return sol