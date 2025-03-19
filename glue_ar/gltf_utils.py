from enum import Enum
import operator
import struct
from typing import Callable, Iterable, List, Literal, Optional, Type, TypeVar, Union

from gltflib import ComponentType, Material, PBRMetallicRoughness

__all__ = [
    "GLTFIndexExportOption",
    "index_export_option",
    "create_material_for_color",
    "add_points_to_bytearray",
    "add_triangles_to_bytearray",
    "index_mins",
    "index_maxes",
]


class GLTFIndexExportOption(Enum):
    Byte = ("B", ComponentType.UNSIGNED_BYTE, 2**8-1)
    Short = ("H", ComponentType.UNSIGNED_SHORT, 2**16-1)
    Int = ("I", ComponentType.UNSIGNED_INT, 2**32-1)

    def __init__(self, format: Literal["B", "H", "I"], component_type: ComponentType, max: int):
        self.format = format
        self.component_type = component_type
        self.max = max


GLTF_COMPRESSION_EXTENSIONS = {
    "draco": "KHR_draco_mesh_compression",
    "meshoptimizer": "EXT_meshopt_compression",
}


def index_export_option(max_index: int) -> GLTFIndexExportOption:
    for option in GLTFIndexExportOption:
        if max_index <= option.max:
            return option
    return GLTFIndexExportOption.Int


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


def add_points_to_bytearray(arr: bytearray,
                            points: Iterable[Iterable[Union[int, float]]],
                            format: Literal["e", "f"] = "f"):
    for point in points:
        for coordinate in point:
            arr.extend(struct.pack(format, coordinate))


def add_triangles_to_bytearray(arr: bytearray,
                               triangles: Iterable[Iterable[int]],
                               export_option: GLTFIndexExportOption = GLTFIndexExportOption.Int):
    for triangle in triangles:
        for index in triangle:
            arr.extend(struct.pack(export_option.format, index))


def add_values_to_bytearray(arr: bytearray,
                            values: Iterable[Union[int, float]],
                            format: Literal["e", "f"] = "f"):
    for value in values:
        arr.extend(struct.pack(format, value))


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
