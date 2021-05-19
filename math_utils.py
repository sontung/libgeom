import numpy as np


def cross(a, b):
    """
    Calculate cross product of 2 vector a and b. a, b is array has shape (3, 1)
    :param a:
    :param b:
    :return:
    """
    a1, a2, a3 = a
    b1, b2, b3 = b
    ss = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
    return np.array(ss)


def dot(a, b):
    """
    Calculate dot product of 2 vector a and b. a, b is array has shape (3, 1)
    :param a:
    :param b:
    :return:
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def ray_intersects_triangle(ray_origin, ray_vector, triangle):
    """
    [optimized]
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
    h = cross(ray_vector, edge2)

    a = dot(edge1, h)

    if -epsilon < a < epsilon:
        return None
    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = cross(s, edge1)
    v = f * dot(ray_vector, q)
    if v < 0.0 or u+v > 1.0:
        return None
    t = f * dot(edge2, q)
    if t > epsilon:
        res = ray_origin + ray_vector * t
        return res
    else:
        return None
