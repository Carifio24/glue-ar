from mcubes import marching_cubes
from numpy import isfinite, linspace

from gltflib import AccessorType, BufferTarget, ComponentType

from glue_vispy_viewers.volume.layer_state import VolumeLayerState
from glue_vispy_viewers.volume.viewer_state import Vispy3DVolumeViewerState

from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.gltf_builder import GLTFBuilder
from glue_ar.common.usd_builder import USDBuilder
from glue_ar.common.volume_export_options import ARIsosurfaceExportOptions
from glue_ar.gltf_utils import add_points_to_bytearray, add_triangles_to_bytearray, \
                               clip_linear_transformations, index_mins, index_maxes
from glue_ar.utils import BoundsWithResolution, frb_for_layer, hex_to_components, isomin_for_layer, \
                          isomax_for_layer, layer_color


@ar_layer_export(VolumeLayerState, "Isosurface", ARIsosurfaceExportOptions, ("gltf", "glb"))
def add_isosurface_layer_gltf(builder: GLTFBuilder,
                              viewer_state: Vispy3DVolumeViewerState,
                              layer_state: VolumeLayerState,
                              options: ARIsosurfaceExportOptions,
                              bounds: BoundsWithResolution):
    data = frb_for_layer(viewer_state, layer_state, bounds)

    isomin = isomin_for_layer(viewer_state, layer_state)
    isomax = isomax_for_layer(viewer_state, layer_state)

    data[~isfinite(data)] = isomin - 10

    levels = linspace(isomin, isomax, num=int(options.isosurface_count))
    opacity = 0.25 * layer_state.alpha
    color = layer_color(layer_state)
    color_components = hex_to_components(color)
    builder.add_material(color_components, opacity=opacity)

    world_bounds = (
        (viewer_state.y_min, viewer_state.y_max),
        (viewer_state.x_min, viewer_state.x_max),
        (viewer_state.z_min, viewer_state.z_max),
    )
    resolution = viewer_state.resolution
    x_range = viewer_state.x_max - viewer_state.x_min
    y_range = viewer_state.y_max - viewer_state.y_min
    z_range = viewer_state.z_max - viewer_state.z_min
    x_spacing = x_range / resolution
    y_spacing = y_range / resolution
    z_spacing = z_range / resolution
    sides = (z_spacing, x_spacing, y_spacing)
    if viewer_state.native_aspect:
        clip_transforms = clip_linear_transformations(world_bounds, clip_size=1)
        clip_sides = [s * transform[0] for s, transform in zip(sides, clip_transforms)]
    else:
        clip_sides = [2 / resolution for _ in range(3)]

    for level in levels[1:]:
        barr = bytearray()
        level_bin = f"layer_{layer_state.layer.uuid}_level_{level}.bin"

        points, triangles = marching_cubes(data, level)
        if len(points) == 0:
            continue

        points = [tuple((-1 + (index + 0.5) * side) for index, side in zip(pt, clip_sides)) for pt in points]
        points = [[p[1], p[0], p[2]] for p in points]
        add_points_to_bytearray(barr, points)
        point_len = len(barr)

        add_triangles_to_bytearray(barr, triangles)
        triangle_len = len(barr) - point_len

        pt_mins = index_mins(points)
        pt_maxes = index_maxes(points)
        tri_mins = [int(min(idx for tri in triangles for idx in tri))]
        tri_maxes = [int(max(idx for tri in triangles for idx in tri))]

        builder.add_buffer(byte_length=len(barr), uri=level_bin)

        buffer = builder.buffer_count - 1
        builder.add_buffer_view(
            buffer=buffer,
            byte_length=point_len,
            byte_offset=0,
            target=BufferTarget.ARRAY_BUFFER,
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.FLOAT,
            count=len(points),
            type=AccessorType.VEC3,
            mins=pt_mins,
            maxes=pt_maxes,
        )
        builder.add_buffer_view(
            buffer=buffer,
            byte_length=triangle_len,
            byte_offset=point_len,
            target=BufferTarget.ELEMENT_ARRAY_BUFFER,
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.UNSIGNED_INT,
            count=len(triangles)*3,
            type=AccessorType.SCALAR,
            mins=tri_mins,
            maxes=tri_maxes,
        )
        builder.add_mesh(
            position_accessor=builder.accessor_count-2,
            indices_accessor=builder.accessor_count-1,
            material=builder.material_count-1,
        )
        builder.add_file_resource(level_bin, data=barr)


@ar_layer_export(VolumeLayerState, "Isosurface", ARIsosurfaceExportOptions, ("usdc", "usda"))
def add_isosurface_layer_usd(
    builder: USDBuilder,
    viewer_state: Vispy3DVolumeViewerState,
    layer_state: VolumeLayerState,
    options: ARIsosurfaceExportOptions,
    bounds: BoundsWithResolution,
):

    data = frb_for_layer(viewer_state, layer_state, bounds)

    isomin = isomin_for_layer(viewer_state, layer_state)
    isomax = isomax_for_layer(viewer_state, layer_state)

    data[~isfinite(data)] = isomin - 10

    isosurface_count = int(options.isosurface_count)
    levels = linspace(isomin, isomax, isosurface_count)
    opacity = layer_state.alpha
    color = layer_color(layer_state)
    color_components = tuple(hex_to_components(color))

    world_bounds = (
        (viewer_state.y_min, viewer_state.y_max),
        (viewer_state.x_min, viewer_state.x_max),
        (viewer_state.z_min, viewer_state.z_max),
    )
    resolution = viewer_state.resolution
    x_range = viewer_state.x_max - viewer_state.x_min
    y_range = viewer_state.y_max - viewer_state.y_min
    z_range = viewer_state.z_max - viewer_state.z_min
    x_spacing = x_range / resolution
    y_spacing = y_range / resolution
    z_spacing = z_range / resolution
    sides = (z_spacing, x_spacing, y_spacing)
    if viewer_state.native_aspect:
        clip_transforms = clip_linear_transformations(world_bounds, clip_size=1)
        clip_sides = [s * transform[0] for s, transform in zip(sides, clip_transforms)]
    else:
        clip_sides = [2 / resolution for _ in range(3)]

    for i, level in enumerate(levels[1:]):
        alpha = (3 * i + isosurface_count) / (4 * isosurface_count) * opacity
        points, triangles = marching_cubes(data, level)
        if len(points) == 0:
            continue

        points = [tuple((-1 + (index + 0.5) * side) for index, side in zip(pt, clip_sides)) for pt in points]
        points = [[p[1], p[0], p[2]] for p in points]
        builder.add_mesh(points, triangles, color_components, alpha)