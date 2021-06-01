import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from mshr import *
import matplotlib.tri as tri
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
class PlotSolutions():
    def __init__(self):
        pass
    
    def create_plot(self, mesh, u, h, func_name, figsize = None):
        n = mesh.num_vertices()
        d = mesh.geometry().dim()
        mesh_coordinates = mesh.coordinates().reshape((n, d))
        triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
        triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
        if figsize is not None:
            fig = plt.figure(figsize=figsize)

        approx_sol = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
        exact_sol = np.asarray([h(cell.midpoint()) for cell in cells(mesh)])
        plt.suptitle(func_name, fontsize=20)
        plt.subplot(121)
        plt.title('Approx solution')
        plt.tripcolor(triangulation, facecolors=approx_sol, edgecolors='k')

        plt.subplot(122)
        plt.title('Exact solution')
        plt.tripcolor(triangulation, facecolors=exact_sol, edgecolors='k')

    def plot_error(self, error_L2, error_C, func_name, figsize=None):

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        plt.suptitle(func_name)
        plt.subplot(121)
        plt.title('Error L2')
        plt.xlabel('Time steps')
        plt.ylabel('Value')
        plt.plot(error_L2)

        plt.subplot(122)
        plt.title('Error max norm')
        plt.xlabel('Time steps')
        plt.ylabel('Value')
        plt.plot(error_C)

    def _animate(self, i):
        self.ax.clear()
        self.ax1.clear()

        k = self.mesh.num_vertices()
        d = self.mesh.geometry().dim()
        mesh_coordinates = self.mesh.coordinates().reshape((k, d))
        triangles = np.asarray([cell.entities(0) for cell in cells(self.mesh)])
        triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
        
        self.ax.set_aspect('equal')
        img = self.ax.tripcolor(triangulation, facecolors=self.zface_u[i] / self.max_u, edgecolors='k', vmin=0, vmax=1)
        self.ax.set_title('Numerical solution')


        self.ax1.set_aspect('equal')
        img1 = self.ax1.tripcolor(triangulation, facecolors=self.zface_u_D[i] / self.max_u_D, edgecolors='k', vmin=0, vmax=1)
        self.ax1.set_title('Exact solution')

        
        return img, img1

    def create_gif(self, zface_u, zface_u_D, mesh, func_name):
        self.zface_u = zface_u
        self.zface_u_D = zface_u_D
        self.mesh = mesh
        plt.style.use('classic')

        self.fig = plt.figure(figsize=(7, 7))
        plt.suptitle(func_name, fontsize=20)
        self.max_u = zface_u.max()
        self.max_u_D = zface_u_D.max()

        self.ax = self.fig.add_subplot(121)

        self.ax1 = self.fig.add_subplot(122)


        anim = animation.FuncAnimation(self.fig, 
                                      self._animate, 
                                      frames=len(zface_u_D),
                                      repeat = True
                                      )
        return anim