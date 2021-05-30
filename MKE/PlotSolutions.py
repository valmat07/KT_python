import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from mshr import *
import matplotlib.tri as tri
class PlotSolutions():
    def __init__(self):
        pass
    def create_plot(self, mesh, u, h, figsize = None):
        n = mesh.num_vertices()
        d = mesh.geometry().dim()
        mesh_coordinates = mesh.coordinates().reshape((n, d))
        triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
        triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        approx_sol = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
        exact_sol = np.asarray([h(cell.midpoint()) for cell in cells(mesh)])
        plt.subplot(121)
        plt.title('Approx solution')
        plt.tripcolor(triangulation, facecolors=approx_sol, edgecolors='k')

        plt.subplot(122)
        plt.title('Exact solution')
        plt.tripcolor(triangulation, facecolors=exact_sol, edgecolors='k')