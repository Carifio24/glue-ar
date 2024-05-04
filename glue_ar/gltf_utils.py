import math
from itertools import product

def sphere_mesh_index(row, column, theta_resolution, phi_resolution):
    if row == 0:
        return 0
    elif row == theta_resolution - 1:
        return (theta_resolution - 2) * phi_resolution + 1
    else:
        return phi_resolution * (row - 1) + column + 1


# theta is the azimuthal angle here
def sphere_mesh(center, radius, theta_resolution=5, phi_resolution=5):
    nonpole_thetas = [i * math.pi / theta_resolution for i in range(1, theta_resolution-1)]
    phis = [i * 2 * math.pi / phi_resolution for i in range(phi_resolution)]
    points = [(
        center[0] + radius * math.cos(phi) * math.sin(theta),
        center[1] + radius * math.sin(phi) * math.sin(theta),
        center[2] + radius * math.cos(theta)
    ) for theta, phi in product(nonpole_thetas, phis)]
    points = [(center[0], center[1], center[2] + radius)] + points + [(center[0], center[1], center[2] - radius)]

    # TODO: Make a cleaner way to handle "modular" aspect of rows
    # Idea: Make column = column % phi_resolution in `sphere_mesh_index` ?
    triangles = [(int(0), i + 1, i) for i in range(1, phi_resolution)]
    tr, pr = theta_resolution, phi_resolution
    triangles.append((0, 1, theta_resolution))
    for row in range(1, theta_resolution - 2):
        for col in range(phi_resolution):
            rc_index = sphere_mesh_index(row, col, tr, pr)
            triangles.append((rc_index, sphere_mesh_index(row+1, col, tr, pr), sphere_mesh_index(row+1, col-1, tr, pr)))
            triangles.append((rc_index, sphere_mesh_index(row, col+1, tr, pr), sphere_mesh_index(row+1, col, tr, pr)))
        triangles.append((sphere_mesh_index(row, pr-1, tr, pr), sphere_mesh_index(row+1, pr-1, tr, pr), sphere_mesh_index(row+1, pr-2, tr, pr)))
        triangles.append((sphere_mesh_index(row, pr-1, tr, pr), sphere_mesh_index(row, 0, tr, pr), sphere_mesh_index(row+1, pr-1, tr, pr)))

    row = theta_resolution - 2
    last_index = sphere_mesh_index(theta_resolution - 1, 0, tr, pr)
    for col in range(phi_resolution-1):
        triangles.append((sphere_mesh_index(row, col, tr, pr), sphere_mesh_index(row, col+1, tr, pr), last_index))
    triangles.append((sphere_mesh_index(row, pr-1, tr, pr), sphere_mesh_index(row, 0, tr, pr), last_index))

    return points, triangles


def offset_triangles(triangle_indices, offset):
    return [[idx + offset for idx in triangle] for triangle in triangle_indices]
