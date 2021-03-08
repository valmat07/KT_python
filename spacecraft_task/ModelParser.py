import numpy as np
class ModelParser():
    def __init__(self, file_name=None):
        super(ModelParser, self).__init__()
        self.file_name = file_name


    def parse(self):
        '''
        return parts_list, normal_list
            parts_list-list of vertices by each part of model
            normals_list-list of normales by each part for model
        '''
        vertexList = []
        finalVertices = []
        parts_list = []
        one_part_list = []
        with open(self.file_name, 'r') as objFile:
            for line in objFile:
                if 'object Part' in line and len(one_part_list) > 0:
                    parts_list.append(one_part_list)
                    one_part_list = []

                split = line.split()

                #if blank line, skip
                if not len(split):
                    continue
                #vertex
                if split[0] == "v":
                    vertexList.append([float(val) for val in split[1:]])
                #faces
                elif split[0] == "f":
                    face = []
                    for idx in split[1:]:
                        face.append(vertexList[int(idx) - 1])
                    one_part_list.append(face)
            #added last part of model
            parts_list.append(one_part_list)
        
        normals_list = []
        for part in parts_list:
            normals_list_part = []
            for surface in part:
                a = np.array(surface[2]) - np.array(surface[0])
                b = np.array(surface[1]) - np.array(surface[0])
                normals_list_part.append((-np.cross(a, b)).tolist())
            normals_list.append(normals_list_part)
        return parts_list, normals_list