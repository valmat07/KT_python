import numpy as np
class ModelParser():
    def __init__(self, file_name=None):
        super(ModelParser, self).__init__()
        self.file_name = file_name


    def parse(self):
        '''
        Parse .obj file and returns lists of vertices and normals.

        Returns:
            parts_list:list of vertices for each part of model
            normals_list:list of normals for each part for model
        '''
        vertexList = []
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
            #add last part of model
            parts_list.append(one_part_list)

        normals_list = []
        for part in parts_list:
            normals_list_part = []
            for surface in part:
                a = np.array(surface[2]) - np.array(surface[0])
                b = np.array(surface[1]) - np.array(surface[0])
                cross_product = -np.cross(a, b)
                normals_list_part.append((cross_product / np.linalg.norm(cross_product)).tolist())
            normals_list.append(normals_list_part)

        return parts_list, normals_list
   
    def _triangle_area(self, a, b, c) :
        a, b, c = np.array(a), np.array(b), np.array(c)
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    
    def _poly_area(self, x, y):
        x, y = np.array(x), np.array(y)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def getAreas(self, parts_list):
        '''
        Returns areas for each part and areas between every two part
        
        Parameters:
            parts_list (lst): list of model parts.
        
        Returns:
            surfaces_area:numpy array containing areas for each part
            surfaces_area_btw_parts:matrix containing intersection areas between parts.
        '''
        surfaces_area = np.zeros(len(parts_list))
        surfaces_area_btw_parts = np.zeros((len(parts_list), len(parts_list)))

        for i, part in enumerate(parts_list):
            for surface in part:
                surfaces_area[i] += self._triangle_area(*surface)
            
            #now if we assume that the intersection is a 2D plane
            #area can be calculated for a polygon
            for j, second_part in enumerate(parts_list[i + 1:]):
                polygon_vert_x, polygon_vert_z = [], []
                tmp_verts = []
                for surface in part:
                    for second_surface in second_part:
                        for vert in surface:
                            for second_vert in second_surface:
                                #according to model if surfaces have same y-cord, means that they intersect 
                                if (vert[1] == second_vert[1])  and (not vert in tmp_verts):
                                    polygon_vert_x.append(vert[0])
                                    polygon_vert_z.append(vert[2])
                                    tmp_verts.append(vert)

                surfaces_area_btw_parts[i, i + j + 1] = self._poly_area(polygon_vert_x, polygon_vert_z)

        surfaces_area_btw_parts[:, 0] = surfaces_area_btw_parts[0, :]
        for i in range(1, len(parts_list)):
            for j in range(i, len(parts_list)):
                surfaces_area_btw_parts[j, i] = surfaces_area_btw_parts[i, j]

        return surfaces_area, surfaces_area_btw_parts