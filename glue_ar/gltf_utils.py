from enum import Enum
import operator
import struct
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Type, TypeVar, Union

from gltflib import AccessorType, ComponentType, Material, PBRMetallicRoughness
from gltflib.gltf import GLTF
from gltflib.gltf_resource import FileResource

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
    Byte = ("B", ComponentType.UNSIGNED_BYTE, 1)
    Short = ("H", ComponentType.UNSIGNED_SHORT, 2)
    Int = ("I", ComponentType.UNSIGNED_INT, 4)

    def __init__(self, format: Literal["B", "H", "I"], component_type: ComponentType, byte_size: int):
        self.format = format
        self.component_type = component_type
        self.byte_size = byte_size

    @property
    def max(self) -> int:
        return (2 ** (8 * self.byte_size)) - 1


GLTF_COMPRESSION_EXTENSIONS = {
    "draco": "KHR_draco_mesh_compression",
    "meshoptimizer": "EXT_meshopt_compression",
}


def byte_size_format(component_type: ComponentType | int) -> Tuple[int, str]:
    match component_type:
        case ComponentType.UNSIGNED_BYTE:
            return 1, "B"
        case ComponentType.UNSIGNED_SHORT:
            return 2, "H"
        case ComponentType.UNSIGNED_INT:
            return 4, "I"
        case ComponentType.FLOAT:
            return 4, "f"
        case _:
            return 0, ""


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


def get_buffer_data(gltf: GLTF, buffer_index: int) -> bytes:
    model = gltf.model
    buffers = model.buffers
    if buffers is None:
        raise ValueError("Model has no buffers!")

    buffer = buffers[buffer_index]
    if buffer.uri is not None:
        resource = gltf.get_resource(uri=buffer.uri)
        if isinstance(resource, FileResource):
            resource.load()
        return resource.data
    else:
        resource = gltf.get_glb_resource()
        return resource.data


def get_indices(gltf: GLTF, mesh_index: int, primitive_index: int = 0):
    model = gltf.model
    if (meshes := model.meshes) is None or \
       (accessors := model.accessors) is None or \
       (buffer_views := model.bufferViews) is None:
        return []

    mesh = meshes[mesh_index]
    primitive = mesh.primitives[primitive_index]
    triangles_accessor_index = primitive.indices

    accessor = accessors[triangles_accessor_index or 0]
    buffer_view_index = accessor.bufferView
    buffer_view = buffer_views[buffer_view_index or 0]
    buffer_index = buffer_view.buffer
    buffer_data = get_buffer_data(gltf, buffer_index)

    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    count = accessor.count
    component_type = accessor.componentType
    data_type = accessor.type

    if data_type != AccessorType.SCALAR.value:
        raise ValueError(f"Indices should be SCALAR! Got {data_type}")

    component_size, format = byte_size_format(component_type)

    index_data = []
    unpack_format = f"<{format}"
    for i in range(count):
        offset = byte_offset + i * component_size
        index_bytes = buffer_data[offset:offset + component_size]
        index = struct.unpack(unpack_format, index_bytes)
        index_data.append(index)
    return index_data


def get_vertex_positions(gltf: GLTF, mesh_index: int, primitive_index: int = 0):
    model = gltf.model
    if (meshes := model.meshes) is None or \
       (accessors := model.accessors) is None or \
       (buffer_views := model.bufferViews) is None:
        return []

    mesh = meshes[mesh_index]
    primitive = mesh.primitives[primitive_index]
    attributes = primitive.attributes
    position_accessor_index = attributes.POSITION

    accessor = accessors[position_accessor_index or 0]
    buffer_view_index = accessor.bufferView
    buffer_view = buffer_views[buffer_view_index or 0]
    buffer_index = buffer_view.buffer
    buffer_data = get_buffer_data(gltf, buffer_index)

    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    count = accessor.count
    component_type = accessor.componentType
    data_type = accessor.type

    if data_type != AccessorType.VEC3.value:
        raise ValueError(f"Vertex positions should be VEC3! Got {data_type}")
    component_size, format = byte_size_format(component_type)
    num_components = 3

    vertex_data = []
    unpack_format = f"<{num_components}{format}"
    for i in range(count):
        offset = byte_offset + i * num_components * component_size
        vertex_bytes = buffer_data[offset:offset + num_components * component_size]
        vertex = struct.unpack(unpack_format, vertex_bytes)
        vertex_data.append(vertex)
    return vertex_data
