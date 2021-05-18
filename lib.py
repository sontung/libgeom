import meshio
import numpy as np
import utils
import marching_cube
import math_utils
import open3d as o3d
from scipy.spatial.distance import euclidean as distance_function
from pykdtree.kdtree import KDTree as Fast_Kd_Tree
from collections import namedtuple


def surface_reconstruct_marching_cube(point_cloud):
    """
    reconstruct the surface of a point cloud using marching cube algorithm
    :param point_cloud:
    :return:
    """
    a_step = 0.1

    bounding_box = point_cloud.get_axis_aligned_bounding_box()
    min_bound = bounding_box.min_bound-a_step
    max_bound = bounding_box.max_bound+a_step

    GridCell = namedtuple('GridCell', ['xyz', 'val'])
    a_tree = Fast_Kd_Tree(np.asarray(point_cloud.points))
    x_steps = np.arange(min_bound[0], max_bound[0], a_step)
    y_steps = np.arange(min_bound[1], max_bound[1], a_step)
    z_steps = np.arange(min_bound[2], max_bound[2], a_step)

    number_cube = x_steps.shape[0] * y_steps.shape[0] * z_steps.shape[0]
    print(min_bound, max_bound, number_cube)

    cube_points2, stu2cnt, stu2cnt_dict = marching_cube.all_points_in_cube(x_steps, y_steps, z_steps, a_step)
    dist_dict, _ = a_tree.query(cube_points2, k=1)
    triangles = []
    a, b, c = len(z_steps)+1, len(y_steps)+1, len(x_steps)+1

    for s, z in enumerate(z_steps):
        for t, y in enumerate(y_steps):
            for u, x in enumerate(x_steps):
                coord = [[x + a_step, y, z], [x, y, z], [x, y + a_step, z], [x + a_step, y + a_step, z],
                         [x + a_step, y, z + a_step], [x, y, z + a_step], [x, y + a_step, z + a_step],
                         [x + a_step, y + a_step, z + a_step]]

                indices = [s*c*b+t*c+(u+1), s*c*b+t*c+u, s*c*b+(t+1)*c+u, s*c*b+(t+1)*c+(u+1),
                           (s+1)*c*b+t*c+(u+1), (s+1)*c*b+t*c+u, (s+1)*c*b+(t+1)*c+u, (s+1)*c*b+(t+1)*c+(u+1)]
                distances = [dist_dict[stu2cnt_dict[indices[0]]], dist_dict[stu2cnt_dict[indices[1]]],
                             dist_dict[stu2cnt_dict[indices[2]]], dist_dict[stu2cnt_dict[indices[3]]],
                             dist_dict[stu2cnt_dict[indices[4]]], dist_dict[stu2cnt_dict[indices[5]]],
                             dist_dict[stu2cnt_dict[indices[6]]], dist_dict[stu2cnt_dict[indices[7]]]]
                cell = GridCell(coord, distances)
                tri, _ = marching_cube.march_cube(cell)
                triangles.extend(tri)
    print("created: ", len(triangles))
    utils.create_obj_file(triangles)


def surface_reconstruct_marching_cube_with_vis(point_cloud):
    """
    reconstruct the surface of a point cloud using marching cube algorithm with visualization
    :param point_cloud:
    :return:
    """
    a_step = 0.1

    bounding_box = point_cloud.get_axis_aligned_bounding_box()
    min_bound = bounding_box.min_bound-a_step
    max_bound = bounding_box.max_bound+a_step

    GridCell = namedtuple('GridCell', ['xyz', 'val'])
    a_tree = Fast_Kd_Tree(np.asarray(point_cloud.points))
    x_steps = np.arange(min_bound[0], max_bound[0], a_step)
    y_steps = np.arange(min_bound[1], max_bound[1], a_step)
    z_steps = np.arange(min_bound[2], max_bound[2], a_step)

    number_cube = x_steps.shape[0] * y_steps.shape[0] * z_steps.shape[0]
    print(min_bound, max_bound, number_cube)

    cube_points2, stu2cnt, stu2cnt_dict = marching_cube.all_points_in_cube(x_steps, y_steps, z_steps, a_step)
    dist_dict, _ = a_tree.query(cube_points2, k=1)
    triangles = []
    a, b, c = len(z_steps)+1, len(y_steps)+1, len(x_steps)+1

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    ctr = vis.get_view_control()
    current_angle = 0

    for s, z in enumerate(z_steps):
        for t, y in enumerate(y_steps):
            for u, x in enumerate(x_steps):
                coord = [[x + a_step, y, z], [x, y, z], [x, y + a_step, z], [x + a_step, y + a_step, z],
                         [x + a_step, y, z + a_step], [x, y, z + a_step], [x, y + a_step, z + a_step],
                         [x + a_step, y + a_step, z + a_step]]

                indices = [s*c*b+t*c+(u+1), s*c*b+t*c+u, s*c*b+(t+1)*c+u, s*c*b+(t+1)*c+(u+1),
                           (s+1)*c*b+t*c+(u+1), (s+1)*c*b+t*c+u, (s+1)*c*b+(t+1)*c+u, (s+1)*c*b+(t+1)*c+(u+1)]
                distances = [dist_dict[stu2cnt_dict[indices[0]]], dist_dict[stu2cnt_dict[indices[1]]],
                             dist_dict[stu2cnt_dict[indices[2]]], dist_dict[stu2cnt_dict[indices[3]]],
                             dist_dict[stu2cnt_dict[indices[4]]], dist_dict[stu2cnt_dict[indices[5]]],
                             dist_dict[stu2cnt_dict[indices[6]]], dist_dict[stu2cnt_dict[indices[7]]]]
                cell = GridCell(coord, distances)
                tri, cube_index = marching_cube.march_cube(cell)

                if len(tri) > 0:
                    triangles.extend(tri)
                    dum = []
                    for _tri in tri:
                        dum.extend(_tri)
                        for geom in utils.visualize_unit_cube(cell, tri):
                            vis.add_geometry(geom)
                            ctr.rotate(current_angle, 0.0)
                else:
                    for geom in utils.visualize_unit_cube(cell, None):
                        vis.add_geometry(geom)
                        ctr.rotate(current_angle, 0.0)
                ctr.rotate(5.0, 0.0)
                current_angle += 1.0

                vis.poll_events()
                vis.update_renderer()

    utils.create_obj_file(triangles)


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
            intersect = math_utils.ray_intersects_triangle(center, vector, triangle)
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
