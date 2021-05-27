import meshio
import numpy as np
import utils
import time
import marching_cube
import math_utils
import open3d as o3d
from scipy.spatial.distance import euclidean as distance_function
from pykdtree.kdtree import KDTree as Fast_Kd_Tree
from collections import namedtuple


def surface_reconstruct_marching_cube(point_cloud, mesh_saved_dir=None, if_vis=False, cube_size=0.75, isovalue=4):
    """
    reconstruct the surface of a point cloud using marching cube algorithm
    :param point_cloud:
    :return:
    """
    a_step = cube_size
    # isovalue = 4

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
                tri, _ = marching_cube.march_cube(cell, isovalue)
                triangles.extend(tri)
    print("created: ", len(triangles))
    if mesh_saved_dir is not None:
        utils.create_obj_file(triangles, mesh_saved_dir)

    # visualization
    if if_vis:
        original_mesh = o3d.io.read_triangle_mesh(mesh_saved_dir)
        vis = o3d.visualization.Visualizer()
        original_mesh_wf = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh)
        original_mesh_wf.paint_uniform_color([0, 0, 0])
        vis.create_window()
        ctr = vis.get_view_control()
        vis.add_geometry(original_mesh_wf)
        while True:
            ctr.rotate(10, 0.0)
            vis.poll_events()
            vis.update_renderer()
    return triangles


def surface_reconstruct_marching_cube_with_vis(point_cloud):
    """
    reconstruct the surface of a point cloud using marching cube algorithm with visualization
    :param point_cloud:
    :return:
    """
    a_step = 5
    isovalue = 4

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
    # vis.add_geometry(point_cloud)
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
                tri, cube_index = marching_cube.march_cube(cell, isovalue)

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
    start = time.time()
    center = np.mean(vertices, axis=0)
    remove_list = {}
    remove_vertices = {}
    keep_vertices = {}
    face_status = {(x, y, z): -1 for x, y, z in faces}  # -1 unknown 0 remove 1 keep
    for i in range(vertices.shape[0]):
        vector = vertices[i] - center
        for x, y, z in faces:
            if face_status[(x, y, z)] >= 0:
                continue
            if remove_vertices.get(x, False) or remove_vertices.get(y, False) or remove_vertices.get(z, False):
                remove_list[(x, y, z)] = 1
                remove_vertices[x] = 1
                remove_vertices[y] = 1
                remove_vertices[z] = 1
                face_status[(x, y, z)] = 0
                continue
            if keep_vertices.get(x, False) or keep_vertices.get(y, False) or keep_vertices.get(z, False):
                face_status[(x, y, z)] = 1
                keep_vertices[x] = 1
                keep_vertices[y] = 1
                keep_vertices[z] = 1
                continue
            if remove_list.get((x, y, z), False):
                continue
            triangle = [vertices[x], vertices[y], vertices[z]]
            intersect = math_utils.ray_intersects_triangle(center, vector, triangle)
            if intersect is not None:
                d1 = distance_function(intersect, center)
                d2 = distance_function(vertices[i], center)
                if round(d1, 5) < round(d2, 5):  # remove
                    print(d1, d2)
                    remove_list[(x, y, z)] = 1
                    remove_vertices[x] = 1
                    remove_vertices[y] = 1
                    remove_vertices[z] = 1
                    face_status[(x, y, z)] = 0
                else:  # keep
                    face_status[(x, y, z)] = 1
                    keep_vertices[x] = 1
                    keep_vertices[y] = 1
                    keep_vertices[z] = 1

        # update faces
        done = True
        for k in face_status:
            if face_status[k] < 0:
                done = False
        if done:
            break

    print("done in", time.time()-start, "removing %d tri" % len(remove_list))
    return remove_list, face_status


def remove_inside_mesh_with_vis(vertices, faces, mesh_dir="test_models/test_sphere.obj"):
    """
    remove the mesh which is inside a bigger mesh
    """
    original_mesh = o3d.io.read_triangle_mesh(mesh_dir)
    vis = o3d.visualization.Visualizer()
    original_mesh_wf = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh)
    original_mesh_wf.paint_uniform_color([0, 0, 1])
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(original_mesh_wf)

    print("processing")
    remove_list, face_status = remove_inside_mesh(vertices, faces)
    print("done processing")

    # write repaired mesh
    kept_face = np.zeros((sum(face_status.values()), 3), dtype=np.int)
    ind = 0
    for k in face_status:
        if face_status[k] == 1:
            kept_face[ind] = k
            ind += 1
    kept_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                          o3d.utility.Vector3iVector(kept_face))
    o3d.io.write_triangle_mesh("test_models/repaired_mesh.obj", kept_mesh)

    # visualize removed mesh
    remove_face = np.zeros((len(remove_list), 3), dtype=np.int)
    for i, k in enumerate(remove_list):
        remove_face[i] = k
    remove_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                            o3d.utility.Vector3iVector(remove_face))
    wf_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(remove_mesh)
    wf_mesh.paint_uniform_color([1, 0, 0])
    vis.add_geometry(wf_mesh)
    ang = 0
    while ang <= 2000:
        ctr.rotate(10.0, 0.0)
        vis.poll_events()
        vis.update_renderer()
        ang += 10
        vis.capture_screen_image("saved/%d.png" % ang)


def loop_subdivision(vertices, triangles, save_name=None, vis=False):
    """
    upsampling mesh using loop subdivision
    :param vertices: np array
    :param triangles: np array
    :param save_name:
    :param vis:
    :return:
    """

    def edge_hash2(p1, p2):
        if p1 < p2:
            return "%d %d" % (p1, p2)
        else:
            return "%d %d" % (p2, p1)

    edge2id = {}
    vert2neighbours = {}
    count = 0
    for x, y, z in triangles:
        if x not in vert2neighbours:
            vert2neighbours[x] = [y, z]
        else:
            vert2neighbours[x].extend([y, z])

        if y not in vert2neighbours:
            vert2neighbours[y] = [x, z]
        else:
            vert2neighbours[y].extend([x, z])

        if z not in vert2neighbours:
            vert2neighbours[z] = [y, x]
        else:
            vert2neighbours[z].extend([y, x])

        # hash edge
        edge_key = edge_hash2(x, y)
        if edge_key not in edge2id:
            edge2id[edge_key] = [count, 0, [z], [x, y]]
            count += 1
        else:
            edge2id[edge_key][2].append(z)
            edge2id[edge_key][1] = 1

        edge_key = edge_hash2(y, z)
        if edge_key not in edge2id:
            edge2id[edge_key] = [count, 0, [x], [y, z]]
            count += 1
        else:
            edge2id[edge_key][2].append(x)
            edge2id[edge_key][1] = 1

        edge_key = edge_hash2(z, x)
        if edge_key not in edge2id:
            edge2id[edge_key] = [count, 0, [y], [z, x]]
            count += 1
        else:
            edge2id[edge_key][2].append(y)
            edge2id[edge_key][1] = 1

    # compute odd vertices
    new_vert_ind = vertices.shape[0]
    new_vert2id = {}
    for edge_key in edge2id:
        edge_id, interior, left_vert, line_vert = edge2id[edge_key]
        if interior == 0:
            vert = vertices[line_vert[0]]*0.5 + vertices[line_vert[1]]*0.5
        elif interior == 1:
            vert = vertices[line_vert[0]]*3/8 + vertices[line_vert[1]]*3/8
            vert += vertices[left_vert[0]]/8 + vertices[left_vert[1]]/8
        edge2id[edge_key].append(new_vert_ind)
        new_vert2id[new_vert_ind] = vert
        new_vert_ind += 1

    # connect faces
    new_faces = []
    for v0, v1, v2 in triangles:
        edge_key1 = edge_hash2(v0, v1)
        edge_key2 = edge_hash2(v1, v2)
        edge_key3 = edge_hash2(v2, v0)
        u8 = edge2id[edge_key1][-1]
        u5 = edge2id[edge_key2][-1]
        u7 = edge2id[edge_key3][-1]

        new_faces.extend([[u8, u5, u7], [u7, v0, u8], [u5, u8, v1], [v2, u7, u5]])

    # compute even vertices
    new_even_vertices = np.zeros_like(vertices)
    for vert_id in vert2neighbours:
        neighbors = list(set(vert2neighbours[vert_id]))
        vert_deg = len(neighbors)
        if vert_deg == 2:
            vert = vertices[vert_id]*0.75 + vertices[neighbors[0]]/8 + vertices[neighbors[1]]/8
        elif vert_deg == 3:
            beta = 0.1875
            arr = np.array(vertices[neighbors]).sum(axis=0)
            vert = vertices[vert_id] * (1 - vert_deg * beta) + arr * beta
        else:
            beta = 0.375/vert_deg
            arr = np.array(vertices[neighbors]).sum(axis=0)
            vert = vertices[vert_id] * (1 - vert_deg * beta) + arr * beta
        new_even_vertices[vert_id] = vert

    triangles2 = np.zeros((len(triangles)+len(new_faces), 3), np.int)
    vertices2 = np.zeros((new_vert_ind, 3), np.float)

    # load into new arrays
    for vid in range(new_vert_ind):
        if vid >= vertices.shape[0]:
            vertices2[vid] = [new_vert2id[vid][0], new_vert2id[vid][1], new_vert2id[vid][2]]
        if vid < vertices.shape[0]:
            vertices2[vid] = [vertices[vid][0], vertices[vid][1], vertices[vid][2]]
    tid = 0
    for x, y, z in triangles:
        triangles2[tid] = [x, y, z]
        tid += 1
    for x, y, z in new_faces:
        triangles2[tid] = [x, y, z]
        tid += 1

    if save_name is not None:
        with open(save_name, 'w') as file:
            for vid in range(new_vert_ind):
                print("v %f %f %f" % (vertices2[vid][0], vertices2[vid][1], vertices2[vid][2]), file=file)
            for x, y, z in triangles2:
                print("f %d %d %d" % (x + 1, y + 1, z + 1), file=file)

    if vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ctr = vis.get_view_control()
        remove_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices2),
                                                o3d.utility.Vector3iVector(triangles2))
        wf_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(remove_mesh)
        vis.add_geometry(wf_mesh)

        ang = 0
        while ang < 2000:
            ctr.rotate(10.0, 0.0)
            vis.poll_events()
            vis.update_renderer()
            ang += 10

    return vertices2, triangles2


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("test_models/airbag.pcd")
    pcd_array = np.asarray(pcd.points)
    # surface_reconstruct_marching_cube_with_vis(pcd)
    # surface_reconstruct_marching_cube(pcd, cube_size=5, isovalue=4)
    mdir = "test_models/airbag.obj"
    a_mesh = meshio.read(mdir)
    remove_inside_mesh_with_vis(a_mesh.points, a_mesh.cells_dict["triangle"], mdir)
