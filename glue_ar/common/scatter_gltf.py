from gltflib import AccessorType, BufferTarget, ComponentType, PrimitiveMode
from glue_vispy_viewers.common.viewer_state import Vispy3DViewerState
from glue_vispy_viewers.scatter.layer_state import ScatterLayerState
from numpy import array, isfinite, ndarray
from numpy.linalg import norm

from typing import List, Literal, Optional, Tuple


from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.scatter_export_options import ARIpyvolumeScatterExportOptions, ARVispyScatterExportOptions
from glue_ar.common.shapes import cone_triangles, cone_points, cylinder_points, cylinder_triangles, \
                                  normalize, rectangular_prism_triangulation, sphere_triangles
from glue_ar.gltf_utils import add_points_to_bytearray, add_triangles_to_bytearray, index_mins, index_maxes
from glue_ar.utils import Viewer3DState, iterable_has_nan, hex_to_components, layer_color, \
                          unique_id, xyz_bounds, xyz_for_layer, Bounds
from glue_ar.common.gltf_builder import GLTFBuilder
from glue_ar.common.scatter import Scatter3DLayerState, ScatterLayerState3D, \
                                   PointsGetter, box_points_getter, IPYVOLUME_POINTS_GETTERS, \
                                   IPYVOLUME_TRIANGLE_GETTERS, VECTOR_OFFSETS, radius_for_scatter_layer, \
                                   scatter_layer_mask, sizes_for_scatter_layer, sphere_points_getter, NoneType

try:
    from glue_jupyter.common.state3d import ViewerState3D
except ImportError:
    ViewerState3D = NoneType


def add_vectors_gltf(builder: GLTFBuilder,
                     viewer_state: Viewer3DState,
                     layer_state: ScatterLayerState3D,
                     data: ndarray,
                     bounds: Bounds,
                     tip_height: float,
                     shaft_radius: float,
                     tip_radius: float,
                     tip_resolution: int = 10,
                     shaft_resolution: int = 10,
                     materials: Optional[List[int]] = None,
                     mask: Optional[ndarray] = None):

    if isinstance(layer_state, ScatterLayerState):
        atts = [layer_state.vx_attribute, layer_state.vy_attribute, layer_state.vz_attribute]
    else:
        atts = [layer_state.vx_att, layer_state.vy_att, layer_state.vz_att]
    vector_data = [layer_state.layer[att].ravel()[mask] for att in atts]

    if viewer_state.native_aspect:
        factor = max((abs(b[1] - b[0]) for b in bounds))
        vector_data = [[0.5 * t / factor for t in v] for v in vector_data]
    else:
        bound_factors = [abs(b[1] - b[0]) for b in bounds]
        vector_data = [[0.5 * t / b for t in v] for v, b in zip(vector_data, bound_factors)]
    vector_data = array(list(zip(*vector_data)))

    offset = VECTOR_OFFSETS[layer_state.vector_origin]
    if layer_state.vector_origin == "tip":
        offset += tip_height

    barr = bytearray()
    triangles = cylinder_triangles(theta_resolution=shaft_resolution)
    triangle_count = len(triangles)
    max_index = max(idx for tri in triangles for idx in tri)
    add_triangles_to_bytearray(barr, triangles)

    if layer_state.vector_arrowhead:
        tip_triangles = cone_triangles(theta_resolution=tip_resolution, start_index=max_index + 1)
        add_triangles_to_bytearray(barr, tip_triangles)
        max_index = max(idx for tri in tip_triangles for idx in tri)
        triangle_count += len(tip_triangles)

    triangles_len = len(barr)
    buffer = builder.buffer_count

    builder.add_buffer_view(
        buffer=buffer,
        byte_length=triangles_len,
        byte_offset=0,
        target=BufferTarget.ELEMENT_ARRAY_BUFFER,
    )
    builder.add_accessor(
        buffer_view=builder.buffer_view_count-1,
        component_type=ComponentType.UNSIGNED_INT,
        count=triangle_count*3,
        type=AccessorType.SCALAR,
        mins=[0],
        maxes=[max_index],
    )
    triangles_accessor = builder.accessor_count - 1

    point_mins = None
    point_maxes = None
    for i, (pt, v) in enumerate(zip(data, vector_data)):
        if iterable_has_nan(v):
            continue
        material_index = materials[i] if materials else builder.material_count - 1
        prev_len = len(barr)
        adjusted_v = v * layer_state.vector_scaling
        length = norm(adjusted_v)
        half_length = 0.5 * length

        adjusted_v = [adjusted_v[i] for i in (1, 2, 0)]
        adjusted_pt = [c + offset * vc for c, vc in zip(pt, adjusted_v)]
        points = cylinder_points(center=adjusted_pt,
                                 radius=shaft_radius,
                                 length=length,
                                 central_axis=adjusted_v,
                                 theta_resolution=shaft_resolution)
        add_points_to_bytearray(barr, points)
        point_count = len(points)
        point_mins = index_mins(points)
        point_maxes = index_maxes(points)

        if layer_state.vector_arrowhead:
            normalized_v = normalize(adjusted_v)
            tip_center_base = [p + half_length * v for p, v in zip(adjusted_pt, normalized_v)]
            tip_points = cone_points(base_center=tip_center_base,
                                     radius=tip_radius,
                                     height=tip_height,
                                     central_axis=adjusted_v,
                                     theta_resolution=tip_resolution)
            add_points_to_bytearray(barr, tip_points)
            point_count += len(tip_points)
            point_mins = index_mins(tip_points, point_mins)
            point_maxes = index_maxes(tip_points, point_maxes)

        builder.add_buffer_view(
            buffer=buffer,
            byte_length=len(barr)-prev_len,
            byte_offset=prev_len,
            target=BufferTarget.ARRAY_BUFFER,
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.FLOAT,
            count=point_count,
            type=AccessorType.VEC3,
            mins=point_mins,
            maxes=point_maxes,
        )

        builder.add_mesh(
            position_accessor=builder.accessor_count-1,
            indices_accessor=triangles_accessor,
            material=material_index,
        )

    uri = f"vectors_{unique_id()}.bin"
    builder.add_buffer(byte_length=len(barr), uri=uri)
    builder.add_file_resource(uri, data=barr)


def add_error_bars_gltf(builder: GLTFBuilder,
                        viewer_state: Viewer3DState,
                        layer_state: ScatterLayerState3D,
                        axis: Literal["x", "y", "z"],
                        data: ndarray,
                        bounds: Bounds,
                        mask: Optional[ndarray] = None):
    att_ending = "attribute" if isinstance(layer_state, ScatterLayerState) else "att"
    att = getattr(layer_state, f"{axis}err_{att_ending}")
    err_values = layer_state.layer[att].ravel()[mask]
    err_values[~isfinite(err_values)] = 0
    index = ['x', 'y', 'z'].index(axis)

    # NB: This ordering is intentional to account for glTF coordinate system
    gltf_index = ['z', 'y', 'x'].index(axis)

    axis_range = abs(bounds[index][1] - bounds[index][0])
    if viewer_state.native_aspect:
        max_range = max((abs(b[1] - b[0]) for b in bounds))
        factor = 1 / max_range
    else:
        factor = 1 / axis_range
    err_values *= factor

    barr = bytearray()

    errors_bin = f"errors_{unique_id()}.bin"
    points = []
    for pt, err in zip(data, err_values):
        start = [c - err if idx == gltf_index else c for idx, c in enumerate(pt)]
        end = [c + err if idx == gltf_index else c for idx, c in enumerate(pt)]
        line_points = (start, end)
        points.extend(line_points)

        add_points_to_bytearray(barr, line_points)

    pt_mins = index_mins(points)
    pt_maxes = index_maxes(points)

    builder.add_buffer(byte_length=len(barr), uri=errors_bin)
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
        mins=pt_mins,
        maxes=pt_maxes,
    )
    builder.add_mesh(
        position_accessor=builder.accessor_count-1,
        material=builder.material_count-1,
        mode=PrimitiveMode.LINES,
    )

    builder.add_file_resource(errors_bin, data=barr)


def add_scatter_layer_gltf(builder: GLTFBuilder,
                           viewer_state: Viewer3DState,
                           layer_state: ScatterLayerState3D,
                           points_getter: PointsGetter,
                           triangles: List[Tuple[int, int, int]],
                           bounds: Bounds,
                           clip_to_bounds: bool = True):
    if layer_state is None:
        return

    bounds = xyz_bounds(viewer_state, with_resolution=False)

    vispy_layer_state = isinstance(layer_state, ScatterLayerState)
    fixed_size = layer_state.size_mode == "Fixed"
    color_mode_attr = "color_mode" if vispy_layer_state else "cmap_mode"
    fixed_color = getattr(layer_state, color_mode_attr, "Fixed") == "Fixed"
    radius = radius_for_scatter_layer(layer_state)
    mask = scatter_layer_mask(viewer_state, layer_state, bounds, clip_to_bounds)

    data = xyz_for_layer(viewer_state, layer_state,
                         preserve_aspect=viewer_state.native_aspect,
                         mask=mask,
                         scaled=True)
    data = data[:, [1, 2, 0]]

    barr = bytearray()
    add_triangles_to_bytearray(barr, triangles)
    triangles_len = len(barr)
    max_index = max(idx for tri in triangles for idx in tri)

    buffer = builder.buffer_count
    builder.add_buffer_view(
        buffer=buffer,
        byte_length=triangles_len,
        byte_offset=0,
        target=BufferTarget.ELEMENT_ARRAY_BUFFER,
    )
    builder.add_accessor(
        buffer_view=builder.buffer_view_count-1,
        component_type=ComponentType.UNSIGNED_INT,
        count=len(triangles)*3,
        type=AccessorType.SCALAR,
        mins=[0],
        maxes=[max_index],
    )
    sphere_triangles_accessor = builder.accessor_count - 1

    first_material_index = builder.material_count
    if fixed_color:
        color = layer_color(layer_state)
        color_components = hex_to_components(color)
        builder.add_material(color=color_components, opacity=layer_state.alpha)

    buffer = builder.buffer_count
    cmap = layer_state.cmap
    cmap_attr = "cmap_attribute" if vispy_layer_state else "cmap_att"
    cmap_att = getattr(layer_state, cmap_attr)
    cmap_vals = layer_state.layer[cmap_att][mask]
    crange = layer_state.cmap_vmax - layer_state.cmap_vmin
    uri = f"layer_{unique_id()}.bin"

    sizes = sizes_for_scatter_layer(layer_state, bounds, mask)
    for i, point in enumerate(data):

        prev_len = len(barr)
        size = radius if fixed_size else sizes[i]
        pts = points_getter(point, size)
        add_points_to_bytearray(barr, pts)
        point_mins = index_mins(pts)
        point_maxes = index_maxes(pts)

        if not fixed_color:
            cval = cmap_vals[i]
            normalized = max(min((cval - layer_state.cmap_vmin) / crange, 1), 0)
            cindex = int(normalized * 255)
            color = cmap(cindex)
            builder.add_material(color, layer_state.alpha)

        builder.add_buffer_view(
            buffer=buffer,
            byte_length=len(barr)-prev_len,
            byte_offset=prev_len,
            target=BufferTarget.ARRAY_BUFFER,
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.FLOAT,
            count=len(pts),
            type=AccessorType.VEC3,
            mins=point_mins,
            maxes=point_maxes,
        )

        material_index = builder.material_count - 1
        builder.add_mesh(
            position_accessor=builder.accessor_count-1,
            indices_accessor=sphere_triangles_accessor,
            material=material_index,
        )

    builder.add_buffer(byte_length=len(barr), uri=uri)
    builder.add_file_resource(uri, data=barr)

    for axis in ("x", "y", "z"):
        if getattr(layer_state, f"{axis}err_visible", False):
            add_error_bars_gltf(
                builder=builder,
                viewer_state=viewer_state,
                layer_state=layer_state,
                axis=axis,
                data=data,
                bounds=bounds,
                mask=mask,
            )

    if layer_state.vector_visible:
        tip_height = radius / 2
        shaft_radius = radius / 8
        tip_radius = tip_height / 2
        if fixed_color:
            materials = None
        else:
            last_material_index = builder.material_count - 1
            materials = list(range(first_material_index, last_material_index+1))
        add_vectors_gltf(
            builder=builder,
            viewer_state=viewer_state,
            layer_state=layer_state,
            data=data,
            bounds=bounds,
            tip_height=tip_height,
            shaft_radius=shaft_radius,
            tip_radius=tip_radius,
            shaft_resolution=10,
            tip_resolution=10,
            materials=materials,
            mask=mask,
        )


@ar_layer_export(ScatterLayerState, "Scatter", ARVispyScatterExportOptions, ("gltf", "glb"))
def add_vispy_scatter_layer_gltf(builder: GLTFBuilder,
                                 viewer_state: Vispy3DViewerState,
                                 layer_state: ScatterLayerState,
                                 options: ARVispyScatterExportOptions,
                                 bounds: Bounds,
                                 clip_to_bounds: bool = True):

    triangles = sphere_triangles(theta_resolution=options.theta_resolution,
                                 phi_resolution=options.phi_resolution)

    points_getter = sphere_points_getter(theta_resolution=options.theta_resolution,
                                         phi_resolution=options.phi_resolution)

    add_scatter_layer_gltf(builder=builder,
                           viewer_state=viewer_state,
                           layer_state=layer_state,
                           points_getter=points_getter,
                           triangles=triangles,
                           bounds=bounds,
                           clip_to_bounds=clip_to_bounds)


@ar_layer_export(Scatter3DLayerState, "Scatter", ARIpyvolumeScatterExportOptions, ("gltf", "glb"))
def add_ipyvolume_scatter_layer_gltf(builder: GLTFBuilder,
                                     viewer_state: ViewerState3D,
                                     layer_state: Scatter3DLayerState,
                                     options: ARIpyvolumeScatterExportOptions,
                                     bounds: Bounds,
                                     clip_to_bounds: bool = True):
    # TODO: What to do for circle2d?
    geometry = str(layer_state.geo)
    triangle_getter = IPYVOLUME_TRIANGLE_GETTERS.get(geometry, rectangular_prism_triangulation)
    triangles = triangle_getter()
    points_getter = IPYVOLUME_POINTS_GETTERS.get(geometry, box_points_getter)

    add_scatter_layer_gltf(builder=builder,
                           viewer_state=viewer_state,
                           layer_state=layer_state,
                           points_getter=points_getter,
                           triangles=triangles,
                           bounds=bounds,
                           clip_to_bounds=clip_to_bounds)
