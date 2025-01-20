from collections import defaultdict
from glue_vispy_viewers.volume.viewer_state import Vispy3DVolumeViewerState
from numpy import isfinite, argwhere, transpose
from typing import Iterable, List, Optional, Union

from glue_vispy_viewers.volume.layer_state import VolumeLayerState

from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.gltf_builder import GLTFBuilder
from glue_ar.common.stl_builder import STLBuilder
from glue_ar.common.usd_builder import USDBuilder
from glue_ar.common.volume_export_options import ARVoxelExportOptions
from glue_ar.usd_utils import material_for_color, sanitize_path
from glue_ar.utils import BoundsWithResolution, alpha_composite, binned_opacity, clamp, \
                          clip_sides, export_label_for_layer, frb_for_layer, hex_to_components, isomin_for_layer, \
                          isomax_for_layer, layer_color, offset_triangles, unique_id, xyz_bounds

from glue_ar.gltf_utils import add_points_to_bytearray, add_triangles_to_bytearray, index_export_option, \
                               index_mins, index_maxes
from glue_ar.common.shapes import rectangular_prism_points, rectangular_prism_triangulation

from gltflib import AccessorType, BufferTarget, ComponentType


@ar_layer_export(VolumeLayerState, "Voxel", ARVoxelExportOptions, ("gltf", "glb"), multiple=True)
def add_voxel_layers_gltf(builder: GLTFBuilder,
                          viewer_state: Vispy3DVolumeViewerState,
                          layer_states: Union[List[VolumeLayerState], VolumeLayerState],
                          options: Iterable[ARVoxelExportOptions],
                          bounds: Optional[BoundsWithResolution] = None,
                          voxels_per_mesh: Optional[int] = None):

    if isinstance(layer_states, VolumeLayerState):
        layer_states = [layer_states]

    if len(layer_states) == 1:
        layer_id = export_label_for_layer(layer_states[0])
    else:
        layer_id = "Voxel Layers"

    bounds = bounds or xyz_bounds(viewer_state, with_resolution=True)
    sides = clip_sides(viewer_state, clip_size=1)
    sides = tuple(sides[i] for i in (1, 2, 0))

    voxels_id = unique_id()
    points_bin = f"points_{voxels_id}.bin"
    triangles_bin = f"triangles_{voxels_id}.bin"

    occupied_voxels = {}

    for layer_state, option in zip(layer_states, options):
        opacity_cutoff = clamp(option.opacity_cutoff, 0, 1)
        opacity_resolution = clamp(option.opacity_resolution, 0, 1)
        opacity_factor = clamp(option.opacity_factor, 0, 2) / 2
        data = frb_for_layer(viewer_state, layer_state, bounds)

        if len(data) == 0:
            continue

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
            opacity = layer_state.alpha * opacity_factor * (value - isomin) / isorange
            adjusted_opacity = binned_opacity(opacity, opacity_resolution)
            indices_tpl = tuple(indices)
            if indices_tpl in occupied_voxels:
                current_color = occupied_voxels[indices_tpl]
                adjusted_a_color = color_components[:3] + [adjusted_opacity]
                new_color = alpha_composite(adjusted_a_color, current_color)
                occupied_voxels[indices_tpl] = new_color
            else:
                occupied_voxels[indices_tpl] = color_components[:3] + [adjusted_opacity]

    # Once we're done doing the alpha compositing, we want to reverse our dictionary setup
    # Right now we have (key, value) as (indices, color)
    # But now we want (color, indices) to do our mesh chunking
    materials_map = {}
    voxels_by_color = defaultdict(list)
    for indices, rgba in occupied_voxels.items():
        if rgba[-1] >= opacity_cutoff:
            rgba = tuple(rgba)
            voxels_by_color[rgba].append(indices)

            if rgba in materials_map:
                material_index = materials_map[rgba]
            else:
                material_index = builder.material_count
                materials_map[rgba] = material_index
                builder.add_material(
                    rgba[:3],
                    rgba[3],
                )

    max_points_per_opacity = max((len(voxels) for voxels in voxels_by_color.values()), default=0)
    if voxels_per_mesh is None:
        voxels_per_mesh = max_points_per_opacity

    tris = []
    triangle_offset = 0
    triangles = rectangular_prism_triangulation()
    pts_count = len(rectangular_prism_points((0, 0, 0), tuple(1 for _ in range(3))))
    voxels_per_mesh = min(voxels_per_mesh, max_points_per_opacity)
    for _ in range(voxels_per_mesh):
        voxel_triangles = offset_triangles(triangles, triangle_offset)
        triangle_offset += pts_count
        tris.append(voxel_triangles)

    triangles_count = len(tris)
    mesh_triangles = [tri for box in tris for tri in box]
    max_triangle_index = max(idx for tri in mesh_triangles for idx in tri)
    index_format = index_export_option(max_triangle_index)

    triangles_barr = bytearray()
    add_triangles_to_bytearray(triangles_barr, mesh_triangles, export_option=index_format)
    triangles_len = len(triangles_barr)

    builder.add_buffer(byte_length=len(triangles_barr), uri=triangles_bin)
    builder.add_file_resource(triangles_bin, data=triangles_barr)

    triangles_buffer = builder.buffer_count - 1
    points_buffer = builder.buffer_count
    builder.add_buffer_view(
        buffer=triangles_buffer,
        byte_length=triangles_len,
        byte_offset=0,
        target=BufferTarget.ELEMENT_ARRAY_BUFFER,
    )

    builder.add_accessor(
        buffer_view=builder.buffer_view_count-1,
        component_type=index_format.component_type,
        count=len(mesh_triangles)*3,
        type=AccessorType.SCALAR,
        mins=[0],
        maxes=[max_triangle_index],
    )

    points_barr = bytearray()
    default_triangles_accessor = builder.accessor_count - 1
    for rgba, voxels in voxels_by_color.items():

        triangles_accessor = default_triangles_accessor
        start = 0
        n_voxels = len(voxels)
        while start < n_voxels:
            mesh_indices = voxels[start:start+voxels_per_mesh]
            points = []
            for indices in mesh_indices:
                center = tuple((-1 + (index + 0.5) * side) for index, side in zip(indices, sides))
                pts = rectangular_prism_points(center, sides)
                points.append(pts)

            prev_ptbarr_len = len(points_barr)
            mesh_points = [c for pt in points for c in pt]
            add_points_to_bytearray(points_barr, mesh_points)
            ptbarr_len = len(points_barr)

            pt_mins = index_mins(mesh_points)
            pt_maxes = index_maxes(mesh_points)

            builder.add_buffer_view(
               buffer=points_buffer,
               byte_length=ptbarr_len-prev_ptbarr_len,
               byte_offset=prev_ptbarr_len,
               target=BufferTarget.ARRAY_BUFFER,
            )
            builder.add_accessor(
                buffer_view=builder.buffer_view_count-1,
                component_type=ComponentType.FLOAT,
                count=len(mesh_points),
                type=AccessorType.VEC3,
                mins=pt_mins,
                maxes=pt_maxes,
            )
            points_accessor = builder.accessor_count - 1

            # This should only happen on the final iteration
            # or not at all, if voxels_per_mesh is a divisor of count
            # But in this case we do need a separate accessor as the
            # byte length is different.
            count = n_voxels - start
            if count < voxels_per_mesh:
                byte_length = count * triangles_len // triangles_count
                last_mesh_triangles = [tri for box in tris[:count] for tri in box]
                max_mesh_triangle_index = max(idx for tri in last_mesh_triangles for idx in tri)
                builder.add_buffer_view(
                    buffer=triangles_buffer,
                    byte_length=byte_length,
                    byte_offset=0,
                    target=BufferTarget.ELEMENT_ARRAY_BUFFER,
                )

                builder.add_accessor(
                    buffer_view=builder.buffer_view_count-1,
                    component_type=index_format.component_type,
                    count=len(last_mesh_triangles)*3,
                    type=AccessorType.SCALAR,
                    mins=[0],
                    maxes=[max_mesh_triangle_index]
                )
                triangles_accessor = builder.accessor_count - 1

            builder.add_mesh(
                layer_id=layer_id,
                position_accessor=points_accessor,
                indices_accessor=triangles_accessor,
                material=materials_map[rgba],
            )
            start += voxels_per_mesh

    builder.add_buffer(byte_length=len(points_barr), uri=points_bin)
    builder.add_file_resource(points_bin, data=points_barr)


@ar_layer_export(VolumeLayerState, "Voxel", ARVoxelExportOptions, ("usda", "usdc", "usdz"), multiple=True)
def add_voxel_layers_usd(builder: USDBuilder,
                         viewer_state: Vispy3DVolumeViewerState,
                         layer_states: Iterable[VolumeLayerState],
                         options: Iterable[ARVoxelExportOptions],
                         bounds: Optional[BoundsWithResolution] = None):

    bounds = bounds or xyz_bounds(viewer_state, with_resolution=True)
    sides = clip_sides(viewer_state, clip_size=1)
    sides = tuple(sides[i] for i in (1, 2, 0))

    triangles = rectangular_prism_triangulation()

    identifier = sanitize_path(f"voxels_{unique_id()}")

    opacity_factor = 1
    occupied_voxels = {}
    colors_map = defaultdict(set)

    for layer_state, option in zip(layer_states, options):
        opacity_cutoff = clamp(option.opacity_cutoff, 0, 1)
        opacity_resolution = clamp(option.opacity_resolution, 0, 1)
        data = frb_for_layer(viewer_state, layer_state, bounds)

        if len(data) == 0:
            continue

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
            adjusted_opacity = binned_opacity(layer_state.alpha * opacity_factor * (value - isomin) / isorange,
                                              opacity_resolution)
            indices_tpl = tuple(indices)
            if indices_tpl in occupied_voxels:
                current_color = occupied_voxels[indices_tpl]
                adjusted_a_color = color_components[:3] + [adjusted_opacity]
                new_color = alpha_composite(adjusted_a_color, current_color)
                occupied_voxels[indices_tpl] = new_color
                colors_map[current_color].remove(indices_tpl)
                colors_map[new_color].add(indices_tpl)
            elif adjusted_opacity >= opacity_cutoff:
                color = color_components[:3] + [adjusted_opacity]
                occupied_voxels[indices_tpl] = color
                colors_map[tuple(color)].add(indices_tpl)

    materials_map = {}

    for rgba, indices_set in colors_map.items():
        if rgba[-1] < opacity_cutoff:
            continue

        if rgba in materials_map:
            material = materials_map[rgba]
        else:
            material = material_for_color(builder.stage, rgba[:3], rgba[3])
            materials_map[rgba] = material
        points = []
        tris = []
        triangle_offset = 0
        for indices in indices_set:
            center = tuple((index + 0.5) * side for index, side in zip(indices, sides))
            pts = rectangular_prism_points(center, sides)
            points.append(pts)
            pt_triangles = offset_triangles(triangles, triangle_offset)
            triangle_offset += len(pts)
            tris.append(pt_triangles)

        mesh_points = [pt for pts in points for pt in pts]
        mesh_triangles = [tri for sphere in tris for tri in sphere]
        builder.add_mesh(mesh_points,
                         mesh_triangles,
                         color=rgba[:3],
                         opacity=rgba[-1],
                         identifier=identifier)

    return builder


@ar_layer_export(VolumeLayerState, "Voxel", ARVoxelExportOptions, ("stl",), multiple=True)
def add_voxel_layers_stl(builder: STLBuilder,
                         viewer_state: Vispy3DVolumeViewerState,
                         layer_states: Iterable[VolumeLayerState],
                         options: Iterable[ARVoxelExportOptions],
                         bounds: Optional[BoundsWithResolution] = None):

    bounds = bounds or xyz_bounds(viewer_state, with_resolution=True)
    sides = clip_sides(viewer_state, clip_size=1)
    sides = tuple(sides[i] for i in (1, 2, 0))

    triangles = rectangular_prism_triangulation()

    opacity_factor = 1
    occupied_voxels = {}

    for layer_state, option in zip(layer_states, options):
        opacity_cutoff = clamp(option.opacity_cutoff, 0, 1)
        opacity_resolution = clamp(option.opacity_resolution, 0, 1)
        data = frb_for_layer(viewer_state, layer_state, bounds)

        if len(data) == 0:
            continue

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
            adjusted_opacity = binned_opacity(layer_state.alpha * opacity_factor * (value - isomin) / isorange,
                                              opacity_resolution)
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

        center = tuple((index + 0.5) * side for index, side in zip(indices, sides))
        points = rectangular_prism_points(center, sides)
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
