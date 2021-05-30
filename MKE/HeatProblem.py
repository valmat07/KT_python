from fenics import *
import numpy as np
from dolfin import *
from mshr import *
class HeatProblem():
    def __init__(self):
        self.radius = 1
        self.amount_fragments = 25
        self.eps = 1e-10
        self.alfa = 1
        self.max_time = 5.0
        self.amount_steps = 25
        self.dt = self.max_time / self.amount_steps
        self.times = [i * self.dt for i in range(self.amount_steps)]

    def solve_problem(self, h_func, g_func, f_func):
        def boundary_D(x, on_boundary):
            if near(x[0]**2 + x[1]**2, self.radius**2, self.eps) and x[1] > 0.0:
                return True
            else:
                return False

        domain = Circle(Point(0, 0), self.radius, self.amount_fragments)
        mesh = generate_mesh(domain, self.amount_fragments, "cgal")

        V = FunctionSpace(mesh, 'P', 1)
        h = Expression(h_func, degree=2, t=0)
        g = Expression(g_func, degree=2, t=0, R=self.radius)
        f = Expression(f_func, alfa = self.alfa, t=0, degree=2)

        bc = DirichletBC(V, h, boundary_D)

        u_n = interpolate(h, V)

        u = TrialFunction(V)
        v = TestFunction(V)

        F = u * v * dx + self.alfa * self.dt * dot(grad(u), grad(v)) * dx - (u_n + self.dt * f) * v * dx - self.alfa*self.dt * g * v * ds
        
        a, L = lhs(F), rhs(F)

        u = Function(V)

        error_L2 = np.zeros(self.amount_steps)
        error_C = np.zeros(self.amount_steps)

        zfaces_u, zfaces_u_D = [], []
        for i in range(self.amount_steps):
            h.t = self.times[i]
            g.t = self.times[i]
            f.t = self.times[i]

            solve(a == L, u, bc)

            u_D = interpolate(h, V)
            error_L2[i] = errornorm(u_D, u, 'L2')
            vertex_values_u_D = u_D.compute_vertex_values(mesh)
            vertex_values_u = u.compute_vertex_values(mesh)
            error_C[i] = np.max(np.abs(vertex_values_u - vertex_values_u_D))

            zfaces_u.append(np.asarray([u(cell.midpoint()) for cell in cells(mesh)]))
            zfaces_u_D.append(np.asarray([u_D(cell.midpoint()) for cell in cells(mesh)]))

            u_n.assign(u)
        
        return error_L2, error_C, np.array(zfaces_u), np.array(zfaces_u_D), mesh