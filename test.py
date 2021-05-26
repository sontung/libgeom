import open3d as o3d
import numpy as np
import utils


def edge_hash(p1, p2):
    mid = (p1+p2)/2
    return "%f %f %f" % (mid[0], mid[1], mid[2])


def edge_hash2(p1, p2):
    if p1 < p2:
        return "%d %d" % (p1, p2)
    else:
        return "%d %d" % (p2, p1)


def loop_subdivision(vertices, triangles, save_name=None, vis=False):
    """
    upsampling mesh using loop subdivision
    :param vertices:
    :param triangles:
    :param save_name:
    :param vis:
    :return:
    """
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
            beta = 3/16.0
            arr = np.array(vertices[neighbors]).sum(axis=0)
            vert = vertices[vert_id] * (1 - vert_deg * beta) + arr * beta
        else:
            beta = 3/8.0/vert_deg
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
    vertices, triangles = utils.read_obj_file_texture_coords("test_models/spot_triangulated.obj")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # ctr = vis.get_view_control()
    # ctr.rotate(180.0, 0.0)

    # for ind in range(100):
    #     mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    #     mesh.compute_vertex_normals()
    #     vis.clear_geometries()
    #     vis.add_geometry(mesh)
    #     ctr.rotate(500.0, 0.0)
    #     vis.capture_screen_image("saved/loop-%d.png" % ind)
    #     vertices, triangles = loop_subdivision(vertices, triangles, "test_models/loop-%d.obj" % ind)
    #     vis.poll_events()
    #     vis.update_renderer()

    import cProfile, io, pstats
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(2):
        vertices, triangles = loop_subdivision(vertices, triangles)
    s = io.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())

