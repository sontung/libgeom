import meshio
import numpy as np
import utils
import math
import open3d as o3d
from scipy.spatial.distance import euclidean as distance_function


def ray_intersects_triangle(ray_origin, ray_vector, triangle):
    """
    Möller–Trumbore ray-triangle intersection algorithm
    (https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)
    :param ray_origin:
    :param ray_vector:
    :param triangle:
    :return:
    """
    epsilon = 0.0000001
    vertex0 = triangle[0]
    vertex1 = triangle[1]
    vertex2 = triangle[2]
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return None
    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_vector, q)
    if v < 0.0 or u+v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    if t > epsilon:
        res = ray_origin + ray_vector * t
        return res
    else:
        return None


def remove_inside_mesh(vertices, faces):
    """
    remove the mesh which is inside a bigger mesh
    :param vertices:
    :param faces:
    :return:
    """
    center = np.mean(vertices, axis=0)
    remove_list = {}

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    current_angle = 0

    for i in range(vertices.shape[0]):
        vector = vertices[i] - center
        for x, y, z in faces:
            if remove_list.get((x, y, z), False):
                continue
            triangle = [vertices[x], vertices[y], vertices[z]]
            intersect = ray_intersects_triangle(center, vector, triangle)
            if intersect is not None:
                d1 = distance_function(intersect, center)
                d2 = distance_function(vertices[i], center)
                if round(d1, 5) < round(d2, 5):
                    remove_list[(x, y, z)] = 1
                    print(d1, d2)
                    for geom in utils.visualize_tri_o3d(triangle):
                        vis.add_geometry(geom)
                        ctr.rotate(current_angle, 0.0)

                ctr.rotate(5.0, 0.0)
                current_angle += 1.0

        if i % 10 == 0:
            vis.poll_events()
            vis.update_renderer()
            print(len(remove_list))


if __name__ == '__main__':
    a_mesh = meshio.read("test_models/test_sphere.obj")
    remove_inside_mesh(a_mesh.points, a_mesh.cells_dict["triangle"])
