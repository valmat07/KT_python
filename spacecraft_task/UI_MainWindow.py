
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
        
        self.parts_list = parts_list
        
        self.glWidget = GL_Widget(parts_list=parts_list, normals_list=normals_list)
        
        h_layout_buttons = QtWidgets.QHBoxLayout()
        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        self.parametrs_file_txt = QtWidgets.QLineEdit()
        self.solution_dir_txt = QtWidgets.QLineEdit()
        self.solution_dir_txt.setEnabled(False)
        self.parametrs_file_txt.setEnabled(False)


        self.btn_select_parametr = QtWidgets.QPushButton("Select parametrs file")
        self.btn_select_parametr.clicked.connect(self.getFile)
        h_layout_buttons.addWidget(self.btn_select_parametr)
        h_layout_buttons.addWidget(self.parametrs_file_txt)

        self.btn_select_dir_solce = QtWidgets.QPushButton("Select directory to save solution")
        self.btn_select_dir_solce.clicked.connect(self.getDir)
        h_layout_buttons.addWidget(self.btn_select_dir_solce)
        h_layout_buttons.addWidget(self.solution_dir_txt)

        self.btn_calc = QtWidgets.QPushButton("Calculate")
        self.btn_calc.clicked.connect(self.calc)
        self.btn_calc.setEnabled(False)


        slider_layout = QtWidgets.QHBoxLayout()
        self.colorbar_txt =  QtWidgets.QLabel()
        self.colorbar_txt.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold))
        self.colorbar_txt.setText('Min value in colorbar: -10 \nMax value in colorbar: 200')
        slider_layout.addWidget(self.colorbar_txt)

        self.spacer = QtWidgets.QSpacerItem(100, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        slider_layout.addItem(self.spacer)

        self.time_txt =  QtWidgets.QLabel()
        self.time_txt.setText('Set maximum time: ')

        slider_layout.addWidget(self.time_txt)

        self.time_value_txt = QtWidgets.QLineEdit()
        slider_layout.addWidget(self.time_value_txt)
        self.time_value_txt.setText('20')

        

        self.slider_txt =  QtWidgets.QLabel()
        self.slider_txt.setText('Temperature slider: ')
        slider_layout.addWidget(self.slider_txt)

        

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.valueChanged.connect(lambda val: self.glWidget.setTempSlider(val))
        self.slider.setEnabled(False)
        slider_layout.addWidget(self.slider)

        v_layout.addLayout(h_layout_buttons)

        v_layout.addLayout(slider_layout)

        h_layout.addWidget(self.glWidget)

        # a figure instance to plot on 
        self.figure = plt.figure() 
        self.canvas = FigureCanvas(self.figure)

       
        #v_layout_inform.addLayout(h_layout_txt)
        #v_layout_inform.addWidget(self.canvas)

        h_layout.addWidget(self.canvas)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.btn_calc)

        self.get_dir_flag = False
        self.get_file_flag = False
        self.setLayout(v_layout)
        self.showMaximized() 

    def calc(self):
        surfaces_area, surfaces_area_btw_parts = self.model_parser.getAreas(self.parts_list)
        self.heat_solver = HeatBalanceSolver(self.parametrs_file_name, surfaces_area, surfaces_area_btw_parts, self.folderpath)
        self.solve = self.heat_solver.solve(0, int(self.time_value_txt.text()))
        self.plot()
        self.glWidget.setTemperature(self.solve)
        QtWidgets.QSlider.setMaximum(self.slider, len(self.solve) - 1)
        self.slider.setEnabled(True)

    def _enableCalc(self):
        #enable calc only if file and dir is selected
        self.btn_calc.setEnabled(self.get_dir_flag & self.get_file_flag)

    def getFile(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
        
        if dlg.exec_():
            filename = dlg.selectedFiles()
            self.parametrs_file_name = filename[0]
            self.get_file_flag = True
            self._enableCalc()
            self.parametrs_file_txt.setText(self.parametrs_file_name.split('/')[-1])

    def getDir(self):
        self.folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.solution_dir_txt.setText(self.folderpath)
        self.get_dir_flag = True
        self._enableCalc()

    def plot(self):
        self.figure.clear() 
        ax = self.figure.add_subplot(111) 
        # plot data 
        for i in range(self.solve.shape[-1]):
            ax.plot(self.solve[:, i], label='element_{}'.format(i + 1)) 
        ax.legend()
        # refresh canvas 
        self.canvas.draw() 