
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, QtWidgets, QtOpenGL
from GL_Widget import GL_Widget
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, parts_list=None, normals_list=None):
        super(Ui_MainWindow, self).__init__()
        self.widget = GL_Widget(parts_list=parts_list, normals_list=normals_list)
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.widget)
        self.setLayout(mainLayout)