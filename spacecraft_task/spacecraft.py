import sys
from ModelParser import ModelParser
from UI_MainWindow import Ui_MainWindow
from PyQt5 import QtCore, QtWidgets, QtOpenGL

if __name__ == '__main__':    
    model_parser = ModelParser('model1.obj')
    
    
    app = QtWidgets.QApplication(sys.argv)    
    Form = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(Form, model_parser=model_parser)    
    ui.show()    
    

    sys.exit(app.exec_())