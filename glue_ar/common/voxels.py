from glue_vispy_viewers.volume.viewer_state import Vispy3DVolumeViewerState
from numpy import isfinite, argwhere, transpose
from typing import Iterable, Optional

from glue_vispy_viewers.volume.layer_state import VolumeLayerState

from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.gltf_builder import GLTFBuilder
from glue_ar.common.stl_builder import STLBuilder
from glue_ar.common.usd_builder import USDBuilder
from glue_ar.common.volume_export_options import ARVoxelExportOptions
from glue_ar.usd_utils import material_for_color
from glue_ar.utils import BoundsWithResolution, alpha_composite, binned_opacity, clamp, clamped_opacity, \
                          frb_for_layer, get_resolution, hex_to_components, isomin_for_layer, \
                          isomax_for_layer, layer_color, unique_id, xyz_bounds

from glue_ar.gltf_utils import add_points_to_bytearray, add_triangles_to_bytearray, \
                               clip_linear_transformations, index_mins, index_maxes
from glue_ar.common.shapes import rectangular_prism_points, rectangular_prism_triangulation

from gltflib import AccessorType, BufferTarget, ComponentType


@ar_layer_export(VolumeLayerState, "Voxel", ARVoxelExportOptions, ("gltf", "glb"), multiple=True)
def add_voxel_layers_gltf(builder: GLTFBuilder,
                          viewer_state: Vispy3DVolumeViewerState,
                          layer_states: Iterable[VolumeLayerState],
                          options: Iterable[ARVoxelExportOptions],
                          bounds: Optional[BoundsWithResolution] = None):

    resolution = get_resolution(viewer_state)
    bounds = bounds or xyz_bounds(viewer_state, with_resolution=True)
    x_range = viewer_state.x_max - viewer_state.x_min
    y_range = viewer_state.y_max - viewer_state.y_min
    z_range = viewer_state.z_max - viewer_state.z_min
    x_spacing = x_range / resolution
    y_spacing = y_range / resolution
    z_spacing = z_range / resolution
    sides = (x_spacing, y_spacing, z_spacing)

    world_bounds = xyz_bounds(viewer_state, with_resolution=False)
    if viewer_state.native_aspect:
        clip_transforms = clip_linear_transformations(world_bounds, clip_size=1)
        clip_sides = [s * transform[0] for s, transform in zip(sides, clip_transforms)]
        clip_sides = [clip_sides[1], clip_sides[2], clip_sides[0]]
    else:
        clip_sides = [2 / resolution for _ in range(3)]

    point_index = 0
    points_barr = bytearray()
    triangles_barr = bytearray()
    voxels_id = unique_id()
    points_bin = f"points_{voxels_id}.bin"
    triangles_bin = f"triangles_{voxels_id}.bin"

    triangles = rectangular_prism_triangulation()
    triangles_barr = bytearray()
    add_triangles_to_bytearray(triangles_barr, triangles)
    triangle_barrlen = len(triangles_barr)

    builder.add_buffer_view(
        buffer=builder.buffer_count+1,
        byte_length=triangle_barrlen,
        byte_offset=0,
        target=BufferTarget.ELEMENT_ARRAY_BUFFER,
    )
    builder.add_accessor(
        buffer_view=builder.buffer_view_count-1,
        component_type=ComponentType.UNSIGNED_INT,
        count=len(triangles) * 3,
        type=AccessorType.SCALAR,
        mins=[0],
        maxes=[7],
    )
    indices_accessor = builder.accessor_count - 1

    opacity_factor = 1
    occupied_voxels = {}

    for layer_state, option in zip(layer_states, options):
        opacity_cutoff = clamp(option.opacity_cutoff, 0, 1)
        opacity_resolution = clamp(option.opacity_resolution, 0, 1)
        data = frb_for_layer(viewer_state, layer_state, bounds)

        isomin = isomin_for_layer(viewer_state, layer_state)
        isomax = isomax_for_layer(viewer_state, layer_state)

        data[~isfinite(data)] = isomin - 1

        data = transpose(data, (1, 0, 2))

        isorange = isomax - isomin
        nonempty_indices = argwhere(data > isomin)

        color = layer_color(layer_state)
        color_components = hex_to_components(color)

        for indices in nonempty_indices:
            value = data[tuple(indices)]
            adjusted_opacity = clamped_opacity(layer_state.alpha * opacity_factor * (value - isomin) / isorange)
            indices_tpl = tuple(indices)
            if indices_tpl in occupied_voxels:
                current_color = occupied_voxels[indices_tpl]
                adjusted_a_color = color_components[:3] + [adjusted_opacity]
                new_color = alpha_composite(adjusted_a_color, current_color)
                occupied_voxels[indices_tpl] = new_color
            else:
                occupied_voxels[indices_tpl] = color_components[:3] + [adjusted_opacity]

    materials_map = {}
    for indices, rgba in occupied_voxels.items():
        if rgba[-1] < opacity_cutoff:
            continue

        center = tuple((-1 + (index + 0.5) * side) for index, side in zip(indices, clip_sides))

        pts = rectangular_prism_points(center, clip_sides)
        prev_ptbarr_len = len(points_barr)
        point_index += len(pts)
        add_points_to_bytearray(points_barr, pts)
        ptbarr_len = len(points_barr)

        pt_mins = index_mins(pts)
        pt_maxes = index_maxes(pts)

        # We're going to use two buffers
        # The first one (index 0) for the points
        # and the second one (index 1) for the triangles
        builder.add_buffer_view(
           buffer=builder.buffer_count,
           byte_length=ptbarr_len-prev_ptbarr_len,
           byte_offset=prev_ptbarr_len,
           target=BufferTarget.ARRAY_BUFFER,
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.FLOAT,
            count=len(pts),
            type=AccessorType.VEC3,
            mins=pt_mins,
            maxes=pt_maxes,
        )
        rgba_tpl = tuple(rgba[:3] + [binned_opacity(rgba[3], opacity_resolution)])
        if rgba_tpl in materials_map:
            material_index = materials_map[rgba_tpl]
        else:
            material_index = builder.material_count
            materials_map[rgba_tpl] = material_index
            builder.add_material(
                rgba[:3],
                rgba[3],
            )
        builder.add_mesh(
            position_accessor=builder.accessor_count-1,
            indices_accessor=indices_accessor,
            material=material_index
        )

    builder.add_buffer(byte_length=len(points_barr), uri=points_bin)
    builder.add_buffer(byte_length=len(triangles_barr), uri=triangles_bin)

    builder.add_file_resource(points_bin, data=points_barr)
    builder.add_file_resource(triangles_bin, data=triangles_barr)


@ar_layer_export(VolumeLayerState, "Voxel", ARVoxelExportOptions, ("usda", "usdc", "usdz"), multiple=True)
def add_voxel_layers_usd(builder: USDBuilder,
                         viewer_state: Vispy3DVolumeViewerState,
                         layer_states: Iterable[VolumeLayerState],
                         options: Iterable[ARVoxelExportOptions],
                         bounds: Optional[BoundsWithResolution] = None):

    resolution = get_resolution(viewer_state)
    bounds = bounds or xyz_bounds(viewer_state, with_resolution=True)
    x_range = viewer_state.x_max - viewer_state.x_min
    y_range = viewer_state.y_max - viewer_state.y_min
    z_range = viewer_state.z_max - viewer_state.z_min
    x_spacing = x_range / resolution
    y_spacing = y_range / resolution
    z_spacing = z_range / resolution
    sides = (x_spacing, y_spacing, z_spacing)

    world_bounds = xyz_bounds(viewer_state, with_resolution=False)
    if viewer_state.native_aspect:
        clip_transforms = clip_linear_transformations(world_bounds, clip_size=1)
        clip_sides = [s * transform[0] for s, transform in zip(sides, clip_transforms)]
        clip_sides = [clip_sides[1], clip_sides[2], clip_sides[0]]
    else:
        clip_sides = [2 / resolution for _ in range(3)]

    triangles = rectangular_prism_triangulation()

    opacity_factor = 1
    occupied_voxels = {}

    for layer_state, option in zip(layer_states, options):
        opacity_cutoff = clamp(option.opacity_cutoff, 0, 1)
        opacity_resolution = clamp(option.opacity_resolution, 0, 1)
        data = frb_for_layer(viewer_state, layer_state, bounds)

        isomin = isomin_for_layer(viewer_state, layer_state)
        isomax = isomax_for_layer(viewer_state, layer_state)

        data[~isfinite(data)] = isomin - 1

        data = transpose(data, (1, 0, 2))

        isorange = isomax - isomin
        nonempty_indices = argwhere(data - isomin > 0)

        color = layer_color(layer_state)
        color_components = hex_to_components(color)

        for indices in nonempty_indices:
            value = data[tuple(indices)]
            adjusted_opacity = clamped_opacity(layer_state.alpha * opacity_factor * (value - isomin) / isorange)
            indices_tpl = tuple(indices)
            if indices_tpl in occupied_voxels:
                current_color = occupied_voxels[indices_tpl]
                adjusted_a_color = color_components[:3] + [adjusted_opacity]
                new_color = alpha_composite(adjusted_a_color, current_color)
                occupied_voxels[indices_tpl] = new_color
            elif adjusted_opacity >= opacity_cutoff:
                occupied_voxels[indices_tpl] = color_components[:3] + [adjusted_opacity]
    materials_map = {}
    mesh = None
    first_point = None
    for indices, rgba in occupied_voxels.items():
        if rgba[-1] < opacity_cutoff:
            continue

        center = tuple((index + 0.5) * side for index, side in zip(indices, clip_sides))
        rgba_tpl = tuple(rgba[:3] + [binned_opacity(rgba[3], opacity_resolution)])
        if rgba_tpl in materials_map:
            material = materials_map[rgba_tpl]
        else:
            material = material_for_color(builder.stage, rgba[:3], rgba[3])
            materials_map[rgba_tpl] = material

        points = rectangular_prism_points(center, clip_sides)
        if mesh is None:
            mesh = builder.add_mesh(points, triangles, rgba[:3], rgba[3])
            first_point = center
        else:
            translation = tuple(p - fp for p, fp in zip(center, first_point))
            builder.add_translated_reference(mesh, translation, material)

    return builder


@ar_layer_export(VolumeLayerState, "Voxel", ARVoxelExportOptions, ("stl",), multiple=True)
def add_voxel_layers_stl(builder: STLBuilder,
                         viewer_state: Vispy3DVolumeViewerState,
                         layer_states: Iterable[VolumeLayerState],
                         options: Iterable[ARVoxelExportOptions],
                         bounds: Optional[BoundsWithResolution] = None):

    resolution = get_resolution(viewer_state)
    bounds = bounds or xyz_bounds(viewer_state, with_resolution=True)
    x_range = viewer_state.x_max - viewer_state.x_min
    y_range = viewer_state.y_max - viewer_state.y_min
    z_range = viewer_state.z_max - viewer_state.z_min
    x_spacing = x_range / resolution
    y_spacing = y_range / resolution
    z_spacing = z_range / resolution
    sides = (x_spacing, y_spacing, z_spacing)

    world_bounds = xyz_bounds(viewer_state, with_resolution=False)
    if viewer_state.native_aspect:
        clip_transforms = clip_linear_transformations(world_bounds, clip_size=1)
        clip_sides = [s * transform[0] for s, transform in zip(sides, clip_transforms)]
        clip_sides = [clip_sides[1], clip_sides[2], clip_sides[0]]
    else:
        clip_sides = [2 / resolution for _ in range(3)]

    triangles = rectangular_prism_triangulation()

    opacity_factor = 1
    occupied_voxels = {}

    for layer_state, option in zip(layer_states, options):
        opacity_cutoff = clamp(option.opacity_cutoff, 0, 1)
        data = frb_for_layer(viewer_state, layer_state, bounds)

        isomin = isomin_for_layer(viewer_state, layer_state)
        isomax = isomax_for_layer(viewer_state, layer_state)

        data[~isfinite(data)] = isomin - 1

        data = transpose(data, (1, 0, 2))

        isorange = isomax - isomin
        nonempty_indices = argwhere(data - isomin > 0)

        color = layer_color(layer_state)
        color_components = hex_to_components(color)

        for indices in nonempty_indices:
            value = data[tuple(indices)]
            adjusted_opacity = clamped_opacity(layer_state.alpha * opacity_factor * (value - isomin) / isorange)
            indices_tpl = tuple(indices)
            if indices_tpl in occupied_voxels:
                current_color = occupied_voxels[indices_tpl]
                adjusted_a_color = color_components[:3] + [adjusted_opacity]
                new_color = alpha_composite(adjusted_a_color, current_color)
                occupied_voxels[indices_tpl] = new_color
            elif adjusted_opacity >= opacity_cutoff:
                occupied_voxels[indices_tpl] = color_components[:3] + [adjusted_opacity]

    for indices, rgba in occupied_voxels.items():
        if rgba[-1] < opacity_cutoff:
            continue

        center = tuple((index + 0.5) * side for index, side in zip(indices, clip_sides))
        points = rectangular_prism_points(center, clip_sides)
        builder.add_mesh(points, triangles)

    return builder


try:
    from glue_jupyter.ipyvolume.volume import VolumeLayerState as IPVVolumeLayerState
    ar_layer_export.add(IPVVolumeLayerState, "Voxel", ARVoxelExportOptions,
                        ("gltf", "glb"), True, add_voxel_layers_gltf)
    ar_layer_export.add(IPVVolumeLayerState, "Voxel", ARVoxelExportOptions,
                        ("usda", "usdc", "usdz"), True, add_voxel_layers_usd)
    ar_layer_export.add(IPVVolumeLayerState, "Voxel", ARVoxelExportOptions,
                        ("stl",), True, add_voxel_layers_stl)
except ImportError:
    pass
