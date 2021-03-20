from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtOpenGL import *
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
class GL_Widget(QGLWidget):
    xRotationChanged = pyqtSignal(int)
    yRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)
    def __init__(self, parent=None, parts_list=None, normals_list=None, temperature=None):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(800, 800)
        self.parts_list = parts_list
        self.normals_list = normals_list
        self.temperature = temperature
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.color = np.zeros(len(parts_list))
        

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

  
        if self.temperature is not None:
            #draw color bar
            glBegin(GL_QUADS)
            x = -5.5
            y = 0
            max_temp_range = self.max_temp.astype(np.int32) + abs(self.min_temp.astype(np.int32))
            for temp in np.arange(0, max_temp_range, max_temp_range/300):
                glColor3f(self._colormap_red(temp/max_temp_range), self._colormap_green(temp/max_temp_range), self._colormap_blue(temp/max_temp_range))
                glVertex3f(x, 0, -4)
                glVertex3f(x, y, -4)
                glVertex3f(x + 0.3, y, -4) 
                glVertex3f(x + 0.3, 0, -4)
                y += 0.01
                
            glEnd()


        gluLookAt(-2.5, 0, -6, -3, 0, 0, 0, 0, 1)
        glTranslatef(-9, 0, 6)
        glRotatef(240, 1, 1, 1)
        glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)

        glPolygonMode(GL_FRONT, GL_FILL)
        glBegin(GL_TRIANGLES)

        for i, (part_vertex, part_normals) in enumerate(zip(self.parts_list, self.normals_list)):
            glColor3f(self._colormap_red(self.color[i]), self._colormap_green(self.color[i]), self._colormap_blue(self.color[i]))
            for surface, surface_normal in zip(part_vertex, part_normals):
                glNormal3fv(surface_normal)
                for vertex in surface:
                    glVertex3fv(vertex)
        glEnd()
        glFlush()
    def _colormap_red(self, x):
        if x < 0.7: 
            return 4.0 * x - 1.5
        else: 
            return -4.0 * x + 4.5
    
    def _colormap_green(self, x):
        if x < 0.5: 
            return 4.0 * x - 0.5
        else: 
            return -4.0 * x + 3.5

    def _colormap_blue(self, x):
        if x < 0.3: 
            return 4.0 * x + 0.5
        else: 
            return -4.0 * x + 2.5

    def setXRotation(self, angle):
        self.normalizeAngle(angle)

        if angle != self.xRot:
            self.xRot = angle
            self.xRotationChanged.emit(angle)
            self.update()

    def setYRotation(self, angle):
        self.normalizeAngle(angle)

        if angle != self.yRot:
            self.yRot = angle
            self.yRotationChanged.emit(angle)
            self.update()

    def setZRotation(self, angle):
        self.normalizeAngle(angle)

        if angle != self.zRot:
            self.zRot = angle
            self.zRotationChanged.emit(angle)
            self.update()
            
    def normalizeAngle(self, angle):
        while (angle < 0):
            angle += 360 * 16

        while (angle > 360 * 16):
            angle -= 360 * 16
            
    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
        elif event.buttons() & Qt.RightButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setZRotation(self.zRot + 8 * dx)

        self.lastPos = event.pos()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()


    def initializeGL(self):
        glClearDepth(1.0)     
        glLight(GL_LIGHT0, GL_POSITION,  (-9, 0, 3, 1)) # point light from the left, top, front
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE )
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()                    
        gluPerspective(70.0, 2, 1, 100.0) 
        glMatrixMode(GL_MODELVIEW)
    
    def setTempSlider(self, val):
        self.color = (self.temperature[val] - self.min_temp) / (self.max_temp - self.min_temp)
        self.update()
 
    def setTemperature(self, temp):
        self.temperature = temp
        self.max_temp = np.max(self.temperature)
        self.min_temp = np.min(self.temperature)
        self.color = (self.temperature[0] - self.min_temp) / (self.max_temp - self.min_temp)
        self.update()
