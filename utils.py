import numpy as np
import open3d as o3d
import sys


def triangles2mesh(triangles):
    vertices = []
    faces = []
    idx = 0
    adict = {}
    for tri in triangles:
        s = "%f %f %f" % (tri[0], tri[1], tri[2])
        if s not in adict:
            adict[s] = idx
            idx += 1
    for s in adict:
        vertices.append(s.split(" "))

    for i in range(0, len(triangles), 3):
        s_face = []
        for peak in triangles[i: i+3]:
            s = "%f %f %f" % (peak[0], peak[1], peak[2])
            s_face.append(s)
        face_numbers = (adict[s_face[0]], adict[s_face[1]], adict[s_face[2]])
        if adict[s_face[0]] != adict[s_face[1]] \
                and adict[s_face[1]] != adict[s_face[2]] \
                and adict[s_face[0]] != adict[s_face[2]]:
            faces.append(face_numbers)
    return np.array(vertices, np.float), np.array(faces, np.int)


def visualize_tri_o3d(p_tri):
    """
    draw a triangle in open3d
    :param p_tri:
    :return:
    """

    lines = []
    for i in range(0, len(p_tri), 3):
        lines.extend([[i, i+1], [i+1, i+2], [i, i+2]])

    line_color = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
    line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector(line_color)
    line_pcd.points = o3d.utility.Vector3dVector(p_tri)

    return [line_pcd]


def create_obj_file(triangles, save_file='model/airbag.obj'):
    """
    save triangles into obj file
    :param triangles: array of vertices, each vertex contains 3d coordinates
    :param save_file:
    :return:
    """
    with open(save_file, 'w') as file:
        idx = 1
        adict = {}
        for tri in triangles:
            s = "v %f %f %f\n" % (tri[0], tri[1], tri[2])
            if s not in adict:
                adict[s] = idx
                idx += 1
        for s in adict:
            print(s, file=file)

        for i in range(0, len(triangles), 3):
            s_face = []
            for peak in triangles[i: i+3]:
                s = "v %f %f %f\n" % (peak[0], peak[1], peak[2])
                s_face.append(s)
            face_numbers = (adict[s_face[0]], adict[s_face[1]], adict[s_face[2]])
            if adict[s_face[0]] != adict[s_face[1]] \
                    and adict[s_face[1]] != adict[s_face[2]] \
                    and adict[s_face[0]] != adict[s_face[2]]:
                f_line = "f %d %d %d\n" % face_numbers
                print(f_line, file=file)


def read_obj_file_texture_coords(filename):
    """
    read an obj file with texture coordinates
    :param filename:
    :return:
    """
    sys.stdin = open(filename, "r")
    lines = sys.stdin.readlines()
    vert = []
    face = []
    for line in lines:
        components = line[:-1].split(" ")
        if components[0] == "v":
            vert.append(components[1:])
        elif components[0] == "f":
            x_, y_, z_ = map(lambda du: int(du.split("/")[0])-1, components[1:])
            face.append([x_, y_, z_])
    return np.array(vert).astype(np.float), np.array(face).astype(np.int)


def enforce_vertices_by_faces(vertices, faces):
    """
    remove unnecessary vertices
    :param vertices:
    :param faces:
    :return:
    """

    triangles = np.zeros((len(faces)*3, 3))
    for i in range(len(faces) * 3):
        triangles[i] = vertices[faces[i//3][i%3]]
    new_vertices, new_face = triangles2mesh(triangles)
    return new_vertices, new_face
