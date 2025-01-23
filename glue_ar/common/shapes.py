from itertools import product
import math
from typing import Iterable, List, Tuple, Union

from numpy import cross, pi

from glue_ar.utils import offset_triangles

__all__ = [
    "rectangular_prism_points",
    "rectangular_prism_triangulation",
    "sphere_mesh_index",
    "sphere_points",
    "sphere_triangles",
    "cylinder_points",
    "cylinder_triangles",
    "cone_points",
    "cone_triangles",
]


def rectangular_prism_points(center: Iterable[float], sides: Iterable[float]) -> List[Tuple[float, float, float]]:
    side_diffs = [(-s / 2, s / 2) for s in sides]
    diffs = product(*side_diffs)
    points = [tuple(c - d for c, d in zip(center, diff)) for diff in diffs]
    return points


def rectangular_prism_triangulation(start_index: int = 0) -> List[Tuple[int, int, int]]:
    triangles = [
        # x = low
        (start_index + 5, start_index + 1, start_index + 7),
        (start_index + 7, start_index + 1, start_index + 3),

        # x = high
        (start_index + 6, start_index + 2, start_index + 4),
        (start_index + 0, start_index + 4, start_index + 2),

        # y = high
        (start_index + 0, start_index + 2, start_index + 1),
        (start_index + 1, start_index + 2, start_index + 3),

        # y = low
        (start_index + 4, start_index + 7, start_index + 6),
        (start_index + 7, start_index + 4, start_index + 5),

        # z = low
        (start_index + 7, start_index + 3, start_index + 2),
        (start_index + 7, start_index + 2, start_index + 6),

        # z = high
        (start_index + 5, start_index + 4, start_index + 1),
        (start_index + 4, start_index + 0, start_index + 1),
    ]

    return triangles


def rectangular_prism_strip(start_index: int = 0) -> Iterable[int]:
    return tuple(start_index + idx for idx in (3, 2, 6, 7, 4, 2, 0, 3, 1, 6, 5, 4, 1, 0))


def sphere_mesh_index(row: int, column: int, theta_resolution: int, phi_resolution: int) -> int:
    if row == 0:
        return 0
    elif row == theta_resolution - 1:
        return (theta_resolution - 2) * phi_resolution + 1
    else:
        column = column % phi_resolution
        return phi_resolution * (row - 1) + column + 1


def sphere_points(center: Union[List[float], Tuple[float, float, float]],
                  radius: float,
                  theta_resolution: int = 5,
                  phi_resolution: int = 5) -> List[Tuple[float, float, float]]:

    # Number of points: phi_resolution * (theta_resolution - 2) + 2
    nonpole_thetas = [i * math.pi / (theta_resolution - 1) for i in range(1, theta_resolution-1)]
    phis = [i * 2 * math.pi / phi_resolution for i in range(phi_resolution)]
    points = [(
        center[0] + radius * math.cos(phi) * math.sin(theta),
        center[1] + radius * math.sin(phi) * math.sin(theta),
        center[2] + radius * math.cos(theta)
    ) for theta, phi in product(nonpole_thetas, phis)]
    points = [(center[0], center[1], center[2] + radius)] + points + [(center[0], center[1], center[2] - radius)]

    return points


def sphere_points_count(theta_resolution: int, phi_resolution: int) -> int:
    return 2 + (theta_resolution - 2) * phi_resolution


def sphere_triangles(theta_resolution: int = 5, phi_resolution: int = 5) -> List[Tuple[int, int, int]]:
    triangles = [(int(0), i, i + 1) for i in range(1, phi_resolution)]
    tr, pr = theta_resolution, phi_resolution
    triangles.append((1, 0, phi_resolution))
    for row in range(1, theta_resolution - 2):
        for col in range(phi_resolution):
            rc_index = sphere_mesh_index(row, col, tr, pr)
            triangles.append((rc_index, sphere_mesh_index(row+1, col-1, tr, pr), sphere_mesh_index(row+1, col, tr, pr)))
            triangles.append((rc_index, sphere_mesh_index(row+1, col, tr, pr), sphere_mesh_index(row, col+1, tr, pr)))

    row = theta_resolution - 2
    last_index = sphere_mesh_index(theta_resolution - 1, 0, tr, pr)
    for col in range(phi_resolution):
        triangles.append((sphere_mesh_index(row, col+1, tr, pr), sphere_mesh_index(row, col, tr, pr), last_index))

    return triangles


def sphere_triangles_count(theta_resolution: int, phi_resolution: int) -> int:
    return 2 * phi_resolution * (theta_resolution - 2)


def normalize(vector: Iterable[float]) -> List[float]:
    magnitude = math.sqrt(sum(c * c for c in vector))
    return [c / magnitude for c in vector]


def orthogonal_basis(vector: List[float]) -> List[List[float]]:
    first = [-vector[1], vector[0] + vector[2], -vector[1]]
    return [
        first,
        [c for c in cross(vector, first)],
    ]


def cylinder_points(center: Iterable[float],
                    radius: float,
                    length: float,
                    central_axis: Iterable[float],
                    theta_resolution: int = 5) -> List[Tuple[float, float, float]]:

    central_axis = normalize(central_axis)
    half_length = length / 2
    endcap_centers = (
        [c - a * half_length for c, a in zip(center, central_axis)],
        [c + a * half_length for c, a in zip(center, central_axis)]
    )

    orthog_1, orthog_2 = orthogonal_basis(central_axis)

    thetas = [2 * pi * i / theta_resolution for i in range(theta_resolution)]

    return [
        tuple(
            c + o1 * radius * math.cos(theta) + o2 * radius * math.sin(theta)
            for c, o1, o2 in zip(center, orthog_1, orthog_2)
        )
        for center, theta in product(endcap_centers, thetas)
    ]


def cylinder_points_count(theta_resolution: int) -> int:
    return 2 * theta_resolution


def cylinder_triangles(theta_resolution: int = 5, start_index: int = 0) -> List[Tuple[int, int, int]]:
    bottom = [
        (0, i + 1, i) for i in range(1, theta_resolution-1)
    ]
    top = [
        (theta_resolution, theta_resolution + i, theta_resolution + i + 1) for i in range(1, theta_resolution-1)
    ]
    bottom_based_sides = [
        (i, (i + 1) % theta_resolution, i + theta_resolution) for i in range(theta_resolution)
    ]
    top_based_sides = [
        (i + theta_resolution, (i + 1) % theta_resolution, (i + 1) % theta_resolution + theta_resolution)
        for i in range(theta_resolution)
    ]
    if start_index > 0:
        bottom = offset_triangles(bottom, start_index)
        top = offset_triangles(top, start_index)
        bottom_based_sides = offset_triangles(bottom_based_sides, start_index)
        top_based_sides = offset_triangles(top_based_sides, start_index)

    return [t for v in (bottom, top, bottom_based_sides, top_based_sides) for t in v]


def cylinder_triangles_count(theta_resolution: int) -> int:
    return 2 * (theta_resolution - 2) + 2 * theta_resolution


def cone_points(base_center: Iterable[float],
                radius: float,
                height: float,
                central_axis: Iterable[float],
                theta_resolution: int = 5) -> List[Tuple[float]]:

    central_axis = normalize(central_axis)
    orthog_1, orthog_2 = orthogonal_basis(central_axis)
    thetas = [2 * pi * i / theta_resolution for i in range(theta_resolution)]
    top = tuple(c + height * a for c, a in zip(base_center, central_axis))

    return [top] + [
        tuple(
            c + o1 * radius * math.cos(theta) + o2 * radius * math.sin(theta)
            for c, o1, o2 in zip(base_center, orthog_1, orthog_2)
        )
        for theta in thetas
    ]


def cone_points_count(theta_resolution: int) -> int:
    return theta_resolution + 1


def cone_triangles(theta_resolution: int = 5, start_index: int = 0) -> List[Tuple[int, int, int]]:
    sides = [(start_index, start_index + i, start_index + 1 + (i % theta_resolution))
             for i in range(1, theta_resolution + 1)]
    bottom = [
        (start_index + i, start_index + 1, 1 + start_index + (i % theta_resolution))
        for i in range(2, theta_resolution)
    ]

    return sides + bottom


def cone_triangles_count(theta_resolution: int) -> int:
    return 2 * theta_resolution - 2
