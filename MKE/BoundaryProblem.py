from fenics import *
import numpy as np
from dolfin import *
from mshr import *
class BoundaryProblem():
    def __init__(self):
        self.radius = 1
        self.amount_fragments = 25
        self.eps = 1e-10
        self.alfa = 1

    def solve_problem(self, h_val, g_val, f_val, txt, formula):
        def boundary_D(x, on_boundary):
            if near(x[0]**2 + x[1]**2, self.radius**2, self.eps) and x[1] > 0.0:
                return True
            else:
                return False

        domain = Circle(Point(0, 0), self.radius, self.amount_fragments)
        mesh = generate_mesh(domain, self.amount_fragments, "cgal")

        V = FunctionSpace(mesh, 'P', 1)
        h = Expression(h_val, degree=2)
        g = Expression(g_val, R = self.radius, degree=2)
        f = Expression(f_val, alfa = self.alfa, degree=2)

        bc = DirichletBC(V, h, boundary_D)

        u = TrialFunction(V)
        v = TestFunction(V)
        
        a = (dot(grad(u), grad(v)) + self.alfa * u * v)*dx
        L = f * v * dx + g * v *ds
        u = Function(V)
        solve(a == L, u, bc)

        error_L2 = errornorm(h, u, 'L2')
        print("L2-error = ", error_L2)

        vertex_values_u_D = h.compute_vertex_values(mesh)
        vertex_values_u = u.compute_vertex_values(mesh)
        error_C = np.max(np.abs(vertex_values_u - vertex_values_u_D))
        print("C-error = ", error_C)

        return mesh, u, h