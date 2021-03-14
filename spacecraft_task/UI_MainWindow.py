
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, QtWidgets, QtOpenGL
from GL_Widget import GL_Widget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib.pyplot as plt 
from HeatBalanceSolver import HeatBalanceSolver
from PyQt5.QtCore import Qt
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, model_parser=None):
        super(Ui_MainWindow, self).__init__()
        self.model_parser = model_parser

        
        parts_list, normals_list = self.model_parser.parse()
        surfaces_area, surfaces_area_btw_parts = self.model_parser.getAreas(parts_list)
        self.heat_solver = HeatBalanceSolver('parametrs.csv', surfaces_area, surfaces_area_btw_parts)
        self.solve = self.heat_solver.solve(0, 20)
        self.glWidget = GL_Widget(parts_list=parts_list, normals_list=normals_list, temperature=self.solve)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        QtWidgets.QSlider.setMaximum(self.slider, len(self.solve) - 1)
        self.slider.valueChanged.connect(lambda val: self.glWidget.setTemp(val))

        v_layout.addWidget(self.slider)

        h_layout.addWidget(self.glWidget)

        # a figure instance to plot on 
        self.figure = plt.figure() 
        self.canvas = FigureCanvas(self.figure) 
        self.plot()
        

        h_layout.addWidget(self.canvas)
        v_layout.addLayout(h_layout)

       
        self.setLayout(v_layout)
        self.showMaximized() 

    def plot(self):
        self.figure.clear() 


        ax = self.figure.add_subplot(111) 
   
        # plot data 
        for i in range(self.solve.shape[-1]):
            ax.plot(self.solve[:, i], label='element_{}'.format(i + 1)) 
        ax.legend()
        # refresh canvas 
        self.canvas.draw() 