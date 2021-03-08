from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtOpenGL import *
class GL_Widget(QGLWidget):
    def __init__(self, parent=None, parts_list=None, normals_list=None):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(1000, 1000)
        self.parts_list = parts_list
        self.normals_list = normals_list

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        gluLookAt(-2.5, 0, -6, -3, 0, 0, 0, 0, 1)
        glTranslatef(-9, 0, 6)
        glRotatef(240, 1, 1, 1)

        glColor3f(1.0, 1.5, 0.0)
        glPolygonMode(GL_FRONT, GL_FILL)
        glBegin(GL_TRIANGLES)

        for part_vertex, part_normals in zip(self.parts_list, self.normals_list):
            for surface, surface_normal in zip(part_vertex, part_normals):
                glNormal3fv(surface_normal)
                for vertex in surface:
                    glVertex3fv(vertex)

        glEnd()
        glFlush()

    def initializeGL(self):
        glClearDepth(1.0)     

        glLight(GL_LIGHT0, GL_POSITION,  (5, 5, 5, 1)) # point light from the left, top, front
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE )

        #glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()                    
        gluPerspective(70.0, 2, 1, 100.0) 
        glMatrixMode(GL_MODELVIEW)

