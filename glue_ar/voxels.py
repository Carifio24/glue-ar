from math import floor
from itertools import product
from numpy import isfinite, linspace
from os.path import join
import operator
import struct

from glue_vispy_viewers.volume.layer_state import VolumeLayerState
from glue_vispy_viewers.volume.viewer_state import Vispy3DVolumeViewerState
from glue_ar.export import compress_gl

from glue_ar.utils import hex_to_components, isomin_for_layer, isomax_for_layer, layer_color

from typing import List 

from gltflib import Accessor, AccessorType, Asset, BufferTarget, BufferView, Primitive, \
    ComponentType, GLTFModel, Node, Scene, Attributes, Mesh, Buffer, \
    Material, PBRMetallicRoughness
from gltflib.gltf import GLTF
from gltflib.gltf_resource import FileResource


def slope_intercept_between(a, b):
    slope = (b[1] - a[1]) / (b[0] - a[0])
    intercept = b[1] - slope * b[0]
    return slope, intercept


def clip_linear_transformations(bounds, clip_size=1):
    ranges = [abs(bds[1] - bds[0]) for bds in bounds]
    max_range = max(ranges)
    line_data = []
    for bds, rg in zip(bounds, ranges):
        frac = rg / max_range
        half_frac = frac / 2
        half_target = clip_size * half_frac
        line_data.append(slope_intercept_between((bds[0], -half_target), (bds[1], half_target)))
    return line_data


def bring_into_clip(points, transforms):
    return [tuple(transform[0] * c + transform[1] for transform, c in zip(transforms, pt)) for pt in points]


def rectangular_prism_mesh(center, sides, start_index=0):
    side_diffs = [(-s / 2, s / 2) for s in sides]
    diffs = product(*side_diffs)
    points = [tuple(c - d for c, d in zip(center, diff)) for diff in diffs]
    triangles = [
        # side
        (start_index + 0, start_index + 2, start_index + 1),
        (start_index + 1, start_index + 2, start_index + 3),

        # side
        (start_index + 4, start_index + 7, start_index + 6),
        (start_index + 7, start_index + 4, start_index + 5),

        # bottom
        (start_index + 7, start_index + 3, start_index + 2),
        (start_index + 7, start_index + 2, start_index + 6),

        # top
        (start_index + 5, start_index + 4, start_index + 1),
        (start_index + 4, start_index + 0, start_index + 1),

        # side
        (start_index + 6, start_index + 2, start_index + 4),
        (start_index + 0, start_index + 4, start_index + 2),

        # side
        (start_index + 5, start_index + 1, start_index + 7),
        (start_index + 7, start_index + 1, start_index + 3),
    ]

    return points, triangles


def offset_triangles(triangle_indices, offset):
    return [[idx + offset for idx in triangle] for triangle in triangle_indices]


def create_material_for_color(
    color: List[int],
    opacity: float
) -> Material:
    return Material(
            pbrMetallicRoughness=PBRMetallicRoughness(
                baseColorFactor=color[:3] + [opacity],
                roughnessFactor=1,
                metallicFactor=0
            ),
            alphaMode="BLEND"
    )


def create_voxel_export(
    viewer_state: Vispy3DVolumeViewerState,
    layer_state: VolumeLayerState,
    precomputed_frbs=None
):
    n_opacities = 16
    color = layer_color(layer_state)
    color_components = hex_to_components(color)
    materials = [create_material_for_color(color_components, i / n_opacities) for i in range(n_opacities + 1)]

    # resolution = int(viewer_state.resolution)
    resolution = 64 
    bounds = [
        (viewer_state.z_min, viewer_state.z_max, resolution),
        (viewer_state.y_min, viewer_state.y_max, resolution),
        (viewer_state.x_min, viewer_state.x_max, resolution)
    ]

    # For now, only consider one layer
    # shape = (resolution, resolution, resolution)
    data = layer_state.layer.compute_fixed_resolution_buffer(
            target_data=layer_state.layer,
            bounds=bounds,
            target_cid=layer_state.attribute)

    isomin = isomin_for_layer(viewer_state, layer_state) 
    isomax = isomax_for_layer(viewer_state, layer_state) 

    data[~isfinite(data)] = isomin - 1

    x_range = viewer_state.x_max - viewer_state.x_min
    y_range = viewer_state.y_max - viewer_state.y_min
    z_range = viewer_state.z_max - viewer_state.z_min
    x_spacing = x_range / resolution
    y_spacing = y_range / resolution
    z_spacing = z_range / resolution
    sides = (x_spacing, y_spacing, z_spacing)

    world_bounds = (
        (viewer_state.x_min, viewer_state.x_max),
        (viewer_state.y_min, viewer_state.y_max),
        (viewer_state.z_min, viewer_state.z_max),
    )
    world_positions = [linspace(bds[0], bds[1], resolution) for bds in world_bounds]
    clip_transforms = clip_linear_transformations(world_bounds, clip_size=1)
    clip_sides = [s * transform[0] for s, transform in zip(sides, clip_transforms)]
    print(clip_sides)

    point_index = 0
    buffer_views = []
    accessors = []
    meshes = []
    points_barr = bytearray()
    triangles_barr = bytearray()
    points_bin = "points.bin"
    triangles_bin = "triangles.bin"
    voxel_count = resolution ** 3

    for indices in product(range(resolution), repeat=3):
        center = tuple((index + 0.5) * side for index, side in zip(indices, clip_sides))

        pts, tris = rectangular_prism_mesh(center, clip_sides, start_index=point_index)
        point_index += len(pts)
        value = data[*indices]
        adjusted_value = (value - isomin) / (isomax - isomin)
        material_index = floor(adjusted_value * n_opacities)

        prev_ptbarr_len = len(points_barr)
        prev_tribarr_len = len(triangles_barr)
        for pt in pts:
            for coord in pt:
                points_barr.extend(struct.pack('f', coord))
        for tri in tris:
            for idx in tri:
                triangles_barr.extend(struct.pack('I', idx))
        ptbarr_len = len(points_barr)
        tribarr_len = len(triangles_barr)

        pt_mins = [min([operator.itemgetter(i)(pt) for pt in pts]) for i in range(3)]
        pt_maxes = [max([operator.itemgetter(i)(pt) for pt in pts]) for i in range(3)]
        tri_mins = [min([operator.itemgetter(i)(tri) for tri in tris]) for i in range(3)]
        tri_maxes = [max([operator.itemgetter(i)(tri) for tri in tris]) for i in range(3)]

        # We're going to use two buffers
        # The first one (index 0) for the points
        # and the second one (index 1) for the triangles
        buffer_views.append(
            BufferView(buffer=0,
                       byteLength=ptbarr_len-prev_ptbarr_len,
                       byteOffset=prev_ptbarr_len,
                       target=BufferTarget.ARRAY_BUFFER.value,
            )
        )
        accessors.append(
            Accessor(bufferView=len(buffer_views)-1,
                     componentType=ComponentType.FLOAT.value,
                     count=len(pts),
                     type=AccessorType.VEC3.value,
                     min=pt_mins,
                     max=pt_maxes,
            )
        )
        buffer_views.append(
            BufferView(buffer=1,
                       byteLength=tribarr_len-prev_tribarr_len,
                       byteOffset=prev_tribarr_len,
                       target=BufferTarget.ELEMENT_ARRAY_BUFFER.value,
            )
        )
        accessors.append(
            Accessor(bufferView=len(buffer_views)-1,
                     componentType=ComponentType.UNSIGNED_INT.value,
                     count=len(tris) * 3,
                     type=AccessorType.SCALAR.value,
                     min=tri_mins,
                     max=tri_maxes,
            )
        )
        meshes.append(
            Mesh(primitives=[
                Primitive(attributes=Attributes(POSITION=len(accessors)-2),
                          indices=len(accessors)-1,
                          material=material_index,
                )]
            )
        )


    points_buffer = Buffer(byteLength=len(points_barr), uri=points_bin)
    triangles_buffer = Buffer(byteLength=len(triangles_barr), uri=triangles_bin)
    buffers = [points_buffer, triangles_buffer]

    nodes = [Node(mesh=i) for i in range(len(meshes))]

    file_resources = [
        FileResource(points_bin, data=points_barr),
        FileResource(triangles_bin, data=triangles_barr),
    ]
    
    node_indices = list(range(len(nodes)))

    model = GLTFModel(
        asset=Asset(version='2.0'),
        scenes=[Scene(nodes=node_indices)],
        nodes=nodes,
        meshes=meshes,
        buffers=buffers,
        bufferViews=buffer_views,
        accessors=accessors,
        materials=materials
    )
    gltf = GLTF(model=model, resources=file_resources)
    filepath = "voxel_test.gltf"
    gltf.export(filepath)
    # print("About to compress")
    # compress_gl(filepath)


def test_prism_mesh():



    center = (0, 0, 0)
    sides = (1, 1, 3)
    points, triangles = rectangular_prism_mesh(center, sides)
    point_mins = [min([operator.itemgetter(i)(point) for point in points]) for i in range(3)]
    point_maxes = [max([operator.itemgetter(i)(point) for point in points]) for i in range(3)]
    
    output_directory = "out"
    
    N_POINTS = 8
    arr = bytearray()
    for point in points:
        for coord in point:
            arr.extend(struct.pack('f', coord))
    
    triangles_offset = len(arr)
    for triangle in triangles:
        for index in triangle:
            arr.extend(struct.pack('I', index))
    
    buffer = Buffer(byteLength=len(arr), uri="buf.bin")
    buffer_views = [
        BufferView(buffer=0, byteLength=triangles_offset, target=BufferTarget.ARRAY_BUFFER.value),
        BufferView(buffer=0, byteOffset=triangles_offset, byteLength=len(arr)-triangles_offset, target=BufferTarget.ELEMENT_ARRAY_BUFFER.value)
    ]
    accessors = [
        Accessor(bufferView=0, componentType=ComponentType.FLOAT.value, count=N_POINTS, type=AccessorType.VEC3.value, min=point_mins, max=point_maxes),
        Accessor(bufferView=1, componentType=ComponentType.UNSIGNED_INT.value, count=len(triangles) * 3, type=AccessorType.SCALAR.value, min=[0], max=[N_POINTS-1])
    ]
    file_resources = [FileResource("buf.bin", data=arr)]
    
    model = GLTFModel(
        asset=Asset(version='2.0'),
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0), indices=1)])],
        buffers=[buffer],
        bufferViews=buffer_views,
        accessors=accessors
    )
    gltf = GLTF(model=model, resources=file_resources)
    gltf.export(join(output_directory, "prism.gltf"))
    gltf.export(join(output_directory, "prism.glb"))


if __name__ == "__main__":
    test_prism_mesh()
