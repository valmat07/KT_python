import pandas as pd
from scipy.integrate import odeint
import numpy as np
from py_expression_eval import Parser
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
        self.A = 100

    def _equation(self, T, t):
        dTdt = []
        for i in range(self.amount_elemnts):
            tmp_eq = 0
            for j in range(self.amount_elemnts):
                k = self.lambdas[i, j] * self.area_btw_surfaces[i, j]
                tmp_eq += k * (T[j] - T[i])
            q_e = -self.epsilon[i] * self.surfaces_area[i] * self.c_0 * ((T[i] / 100) ** 4)

            if i == 1:
                tmp_eq += q_e + self.A*(20 + 3 * np.cos(t/4))

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

        for value in parametrs_df['q_r']:
            self.q_r.append(value)


    def get_stationary_solution(self):
        return fsolve(self._equation, np.zeros(self.amount_elemnts), args=(0.0))

    def solve(self, t0, t1):
        t = np.linspace(t0, t1)
        init_cond =  np.zeros(5)#self.get_stationary_solution()
        sol = odeint(self._equation, init_cond, t)
        return sol