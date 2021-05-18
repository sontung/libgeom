import numpy as np


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
