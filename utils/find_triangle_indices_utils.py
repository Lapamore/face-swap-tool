import numpy as np

def find_triangle_indices(triangle_list, original_landmarks):
    triangles_indeces = []

    for triangle_points in triangle_list:
        pt1 = (int(triangle_points[0]), int(triangle_points[1]))
        pt2 = (int(triangle_points[2]), int(triangle_points[3]))
        pt3 = (int(triangle_points[4]), int(triangle_points[5]))

        index_pt1 = np.argwhere((original_landmarks == pt1).all(axis=1)).flatten()[0]
        index_pt2 = np.argwhere((original_landmarks == pt2).all(axis=1)).flatten()[0]
        index_pt3 = np.argwhere((original_landmarks == pt3).all(axis=1)).flatten()[0]

        if (index_pt1 is not None) and (index_pt2 is not None) and (index_pt3 is not None):
                triangle = (index_pt1, index_pt2, index_pt3)
                triangles_indeces.append(triangle)

    return triangles_indeces
