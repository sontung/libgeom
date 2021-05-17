import numpy as np
import open3d as o3d


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
