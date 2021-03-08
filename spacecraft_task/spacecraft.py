import sys
from ModelParser import ModelParser
from UI_MainWindow import Ui_MainWindow
from PyQt5 import QtCore, QtWidgets, QtOpenGL

if __name__ == '__main__':    
    model_parser = ModelParser('model1.obj')
    parts_list, normals_list = model_parser.parse()
    
    app = QtWidgets.QApplication(sys.argv)    
    Form = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(Form, parts_list = parts_list, normals_list = normals_list)    
    ui.show()    
    

    sys.exit(app.exec_())