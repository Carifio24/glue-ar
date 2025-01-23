import operator
import struct
from typing import Callable, Iterable, List, Optional, Type, TypeVar, Union

from gltflib import Material, PBRMetallicRoughness

__all__ = [
    "create_material_for_color",
    "add_points_to_bytearray",
    "add_triangles_to_bytearray",
    "index_mins",
    "index_maxes",
]


GLTF_COMPRESSION_EXTENSIONS = {
    "draco": "KHR_draco_mesh_compression",
    "meshoptimizer": "EXT_meshopt_compression",
}

SHORT_MAX = 65_535


def create_material_for_color(
    color: List[int],
    opacity: float
) -> Material:
    rgb = [t / 256 for t in color[:3]]
    return Material(
            pbrMetallicRoughness=PBRMetallicRoughness(
                baseColorFactor=rgb + [opacity],
                roughnessFactor=1,
                metallicFactor=0
            ),
            alphaMode="BLEND"
    )


def add_points_to_bytearray(arr: bytearray, points: Iterable[Iterable[Union[int, float]]]):
    for point in points:
        for coordinate in point:
            arr.extend(struct.pack('f', coordinate))


def add_triangles_to_bytearray(arr: bytearray,
                               triangles: Iterable[Iterable[int]],
                               short: bool = False):
    format = "H" if short else "I"
    for triangle in triangles:
        for index in triangle:
            arr.extend(struct.pack(format, index))


def add_indices_to_bytearray(arr: bytearray,
                             indices: Iterable[int],
                             short: bool = False):
    format = "H" if short else "I"
    for index in indices:
        arr.extend(struct.pack(format, index))


T = TypeVar("T", bound=Union[int, float])


def index_extrema(items: List[List[T]],
                  extremum: Callable[[T, T], T],
                  previous: Optional[List[List[T]]] = None,
                  type: Type[T] = float) -> List[List[T]]:
    size = len(items[0])
    extrema = [type(extremum([operator.itemgetter(i)(item) for item in items])) for i in range(size)]
    if previous is not None:
        extrema = [extremum(x, p) for x, p in zip(extrema, previous)]
    return extrema


def index_mins(items, previous=None, type: Type[T] = float) -> List[List[T]]:
    return index_extrema(items, extremum=min, type=type, previous=previous)


def index_maxes(items, previous=None, type: Type[T] = float) -> List[List[T]]:
    return index_extrema(items, extremum=max, type=type, previous=previous)
