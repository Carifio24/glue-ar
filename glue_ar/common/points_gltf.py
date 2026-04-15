from collections import defaultdict
from numpy import argwhere, isfinite
from gltflib import AccessorType, BufferTarget, ComponentType, PrimitiveMode
from math import ceil
from typing import List, Union
from glue_vispy_viewers.common.viewer_state import Vispy3DViewerState
from glue_vispy_viewers.scatter.layer_state import ScatterLayerState
from glue_vispy_viewers.volume.layer_state import VolumeLayerState

from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.scatter import Scatter3DLayerState, ScatterLayerState3D, scatter_layer_mask
from glue_ar.gltf_utils import add_points_to_bytearray, index_mins, index_maxes
from glue_ar.utils import Bounds, Viewer3DState, alpha_composite, binned_opacity, clamp_with_resolution, clip_sides, color_identifier, frb_for_layer, hex_to_components, \
        isomin_for_layer, isomax_for_layer, layer_color, unique_id, xyz_bounds, xyz_for_layer
from glue_ar.common.gltf_builder import GLTFBuilder
from glue_ar.common.scatter_export_options import ARPointExportOptions

try:
    from glue_jupyter.common.state3d import ViewerState3D
except ImportError:
    ViewerState3D = None


def add_scatter_points_layer_gltf(builder: GLTFBuilder,
                                  viewer_state: Viewer3DState,
                                  layer_state: ScatterLayerState3D,
                                  bounds: Bounds,
                                  clip_to_bounds: bool = True):

    if layer_state is None:
        return

    layer_id = "Point Layers"

    bounds = xyz_bounds(viewer_state, with_resolution=False)

    vispy_layer_state = isinstance(layer_state, ScatterLayerState)
    color_mode_attr = "color_mode" if vispy_layer_state else "cmap_mode"
    fixed_color = getattr(layer_state, color_mode_attr, "Fixed") == "Fixed"

    mask = scatter_layer_mask(viewer_state, layer_state, bounds, clip_to_bounds)
    data = xyz_for_layer(viewer_state, layer_state,
                         preserve_aspect=viewer_state.native_aspect,
                         mask=mask,
                         scaled=True)
    data = data[:, [1, 2, 0]]

    uri = f"layer_{unique_id()}.bin"

    if fixed_color:
        color = layer_color(layer_state)
        color_components = hex_to_components(color)
        builder.add_material(color=color_components, opacity=layer_state.alpha)

        barr = bytearray()
        add_points_to_bytearray(barr, data)

        data_mins = index_mins(data)
        data_maxes = index_maxes(data)

        builder.add_buffer(byte_length=len(barr), uri=uri)
        builder.add_buffer_view(
            buffer=builder.buffer_count-1,
            byte_length=len(barr),
            byte_offset=0,
            target=BufferTarget.ARRAY_BUFFER
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.FLOAT,
            count=len(data),
            type=AccessorType.VEC3,
            mins=data_mins,
            maxes=data_maxes,
        )
        builder.add_mesh(
            layer_id=layer_id,
            position_accessor=builder.accessor_count-1,
            material=builder.material_count-1,
            mode=PrimitiveMode.POINTS
        )
        builder.add_file_resource(uri, data=barr)
    else:
        # If we don't have fixed colors, the idea is to make a different "mesh" for each different color used
        # So first we need to run through the points and determine which color they have, and group ones with
        # the same color together
        points_by_color = defaultdict(list)
        cmap = layer_state.cmap
        cmap_attr = "cmap_attribute" if vispy_layer_state else "cmap_att"
        cmap_att = getattr(layer_state, cmap_attr)
        cmap_vals = layer_state.layer[cmap_att][mask]
        crange = layer_state.cmap_vmax - layer_state.cmap_vmin
        opacity = layer_state.alpha

        for i, point in enumerate(data):
            cval = cmap_vals[i]
            normalized = max(min((cval - layer_state.cmap_vmin) / crange, 1), 0)
            cindex = int(normalized * 255)
            color = cmap(cindex)
            points_by_color[color].append(point)

        for color, points in points_by_color.items():
            builder.add_material(color, opacity)
            material_index = builder.material_count - 1

            uri = f"layer_{unique_id()}_{color_identifier(color, opacity)}"

            barr = bytearray()
            add_points_to_bytearray(barr, points)
            point_mins = index_mins(points)
            point_maxes = index_maxes(points)

            builder.add_buffer(byte_length=len(barr), uri=uri)
            builder.add_buffer_view(
                buffer=builder.buffer_count-1,
                byte_length=len(barr),
                byte_offset=0,
                target=BufferTarget.ARRAY_BUFFER
            )
            builder.add_accessor(
                buffer_view=builder.buffer_view_count-1,
                component_type=ComponentType.FLOAT,
                count=len(points),
                type=AccessorType.VEC3,
                mins=point_mins,
                maxes=point_maxes
            )
            builder.add_mesh(
                position_accessor=builder.accessor_count-1,
                material=material_index,
                mode=PrimitiveMode.POINTS,
            )
            builder.add_file_resource(uri, data=barr)


def add_volume_points_layer_gltf(builder: GLTFBuilder,
                                 viewer_state: Viewer3DState,
                                 layer_states: List[VolumeLayerState],
                                 bounds: Bounds):
    if not layer_states:
        return

    bounds = bounds or xyz_bounds(viewer_state, with_resolution=False)
    sides = clip_sides(viewer_state, clip_size=1)
    sides = tuple(sides[i] for i in (1, 2, 0))

    occupied_points = {}

    layer_id = "Point Layers"

    for layer_state in layer_states:
        data = frb_for_layer(viewer_state, layer_state, bounds)
        isomin = isomin_for_layer(viewer_state, layer_state)
        isomax = isomax_for_layer(viewer_state, layer_state)

        data[~isfinite(data)] = isomin - 1

        isorange = isomax - isomin
        nonempty_indices = argwhere(data > isomin)

        # TODO: Dummy values for now
        cmap_resolution = 0.01
        opacity_cutoff = 0.01
        opacity_factor = 1

        color = layer_color(layer_state)
        color_components = hex_to_components(color)
        if layer_state.color_mode == "Linear":
            voxel_colors = layer_state.cmap([i * cmap_resolution for i in range(ceil(1 / cmap_resolution) + 1)])
            voxel_colors = [[int(256 * float(c)) for c in vc[:3]] for vc in voxel_colors]

        for indices in nonempty_indices:
            value = data[tuple(indices)]
            t_voxel = (value - isomin) / isorange
            if t_voxel > 0 and hasattr(layer_state, 'stretch'):
                t_voxel = layer_state.stretch_object([t_voxel], **layer_state.stretch_parameters)[0]
            t_voxel = clamp_with_resolution(t_voxel, 0, 1, cmap_resolution)
            adjusted_opacity = binned_opacity(layer_state.alpha * opacity_factor * t_voxel, cmap_resolution)

            if adjusted_opacity == 0:
               continue 
            
            if layer_state.color_mode == "Fixed":
                voxel_color_components = color_components
            else:
                index = round(t_voxel / cmap_resolution)
                voxel_color_components = voxel_colors[index]

            indices_tpl = tuple(indices)
            adjusted_a_color = voxel_color_components[:3] + [adjusted_opacity]
            if indices_tpl in occupied_points:
                current_color = occupied_points[indices_tpl]
                new_color = alpha_composite(adjusted_a_color, current_color)
                occupied_points[indices_tpl] = new_color
            else:
                occupied_points[indices_tpl] = adjusted_a_color

    # Once we're done doing the alpha compositing, we want to reverse our dictionary setup
    # Right now we have (key, value) as (indices, color)
    # But now we want (color, indices) to do our mesh chunking
    materials_map = {}
    points_by_color = defaultdict(list)
    for indices, rgba in occupied_points.items():
        if rgba[-1] >= opacity_cutoff:
            rgba = tuple(rgba)
            points_by_color[rgba].append(indices)

            if rgba in materials_map:
                material_index = materials_map[rgba]
            else:
                material_index = builder.material_count
                materials_map[rgba] = material_index
                builder.add_material(
                    rgba[:3],
                    rgba[3],
                )

    for rgba, points in points_by_color.items():

        uri = f"layer_{unique_id()}_{color_identifier(rgba[:3], rgba[3])}"
        barr = bytearray()
        add_points_to_bytearray(barr, points)
        point_mins = index_mins(points)
        point_maxes = index_maxes(points)

        builder.add_buffer(byte_length=len(barr), uri=uri)
        builder.add_buffer_view(
            buffer=builder.buffer_count-1,
            byte_length=len(barr),
            byte_offset=0,
            target=BufferTarget.ARRAY_BUFFER,
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.FLOAT,
            count=len(points),
            type=AccessorType.VEC3,
            mins=point_mins,
            maxes=point_maxes
        )
        builder.add_mesh(
            layer_id=layer_id,
            position_accessor=builder.accessor_count-1,
            material=materials_map[rgba],
            mode=PrimitiveMode.POINTS,
        )
        builder.add_file_resource(uri, data=barr)


@ar_layer_export(VolumeLayerState, "Points", ARPointExportOptions, ("gltf", "glb"))
def add_vispy_volume_points_layer_gltf(builder: GLTFBuilder,
                                       viewer_state: Vispy3DViewerState,
                                       layer_state: ScatterLayerState,
                                       options: ARPointExportOptions,
                                       bounds: Bounds):
    add_volume_points_layer_gltf(builder=builder,
                                 viewer_state=viewer_state,
                                 layer_states=[layer_state],
                                 bounds=bounds)


@ar_layer_export(ScatterLayerState, "Points", ARPointExportOptions, ("gltf", "glb"))
def add_vispy_points_layer_gltf(builder: GLTFBuilder,
                                viewer_state: Vispy3DViewerState,
                                layer_state: ScatterLayerState,
                                options: ARPointExportOptions,
                                bounds: Bounds,
                                clip_to_bounds: bool = True):
    add_scatter_points_layer_gltf(builder=builder,
                                  viewer_state=viewer_state,
                                  layer_state=layer_state,
                                  bounds=bounds,
                                  clip_to_bounds=clip_to_bounds)


@ar_layer_export(Scatter3DLayerState, "Points", ARPointExportOptions, ("gltf", "glb"))
def add_ipyvolume_points_layer_gltf(builder: GLTFBuilder,
                                    viewer_state: ViewerState3D,
                                    layer_state: ScatterLayerState,
                                    options: ARPointExportOptions,
                                    bounds: Bounds,
                                    clip_to_bounds: bool = True):
    add_scatter_points_layer_gltf(builder=builder,
                                  viewer_state=viewer_state,
                                  layer_state=layer_state,
                                  bounds=bounds,
                                  clip_to_bounds=clip_to_bounds)
