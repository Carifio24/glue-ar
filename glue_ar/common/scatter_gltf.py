from gltflib import AccessorType, BufferTarget, ComponentType, PrimitiveMode
from glue.utils import ensure_numerical
from glue_vispy_viewers.scatter.layer_state import ScatterLayerState
from glue_vispy_viewers.scatter.viewer_state import Vispy3DScatterViewerState
from glue_vispy_viewers.volume.viewer_state import Vispy3DViewerState
from numpy import array, clip, isfinite, isnan, ndarray, ones, sqrt
from numpy.linalg import norm

from typing import List, Literal, Optional

from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.scatter import radius_for_scatter_layer, VECTOR_OFFSETS
from glue_ar.common.scatter_export_options import ARScatterExportOptions
from glue_ar.common.shapes import cone_triangles, cone_points, cylinder_points, cylinder_triangles, \
                                  normalize, sphere_points, sphere_triangles
from glue_ar.gltf_utils import add_points_to_bytearray, add_triangles_to_bytearray, index_mins, index_maxes
from glue_ar.utils import iterable_has_nan, hex_to_components, layer_color, mask_for_bounds, \
                          unique_id, xyz_bounds, xyz_for_layer, Bounds
from glue_ar.common.gltf_builder import GLTFBuilder


def add_vectors_gltf(builder: GLTFBuilder,
                     viewer_state: Vispy3DViewerState,
                     layer_state: ScatterLayerState,
                     data: ndarray,
                     bounds: Bounds,
                     tip_height: float,
                     shaft_radius: float,
                     tip_radius: float,
                     tip_resolution: int = 10,
                     shaft_resolution: int = 10,
                     materials: Optional[List[int]] = None,
                     mask: Optional[ndarray] = None):

    atts = [layer_state.vx_attribute, layer_state.vy_attribute, layer_state.vz_attribute]
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
                        viewer_state: Vispy3DViewerState,
                        layer_state: ScatterLayerState,
                        axis: Literal["x", "y", "z"],
                        data: ndarray,
                        bounds: Bounds,
                        mask: Optional[ndarray] = None):
    att = getattr(layer_state, f"{axis}err_attribute")
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


@ar_layer_export(ScatterLayerState, "Scatter", ARScatterExportOptions, ("gltf", "glb"))
def add_scatter_layer_gltf(builder: GLTFBuilder,
                           viewer_state: Vispy3DScatterViewerState,
                           layer_state: ScatterLayerState,
                           options: ARScatterExportOptions,
                           bounds: Bounds,
                           clip_to_bounds: bool = True):
    theta_resolution = options.theta_resolution
    phi_resolution = options.phi_resolution
    bounds = xyz_bounds(viewer_state, with_resolution=False)
    if clip_to_bounds:
        mask = mask_for_bounds(viewer_state, layer_state, bounds)
    else:
        mask = None

    fixed_color = layer_state.color_mode == "Fixed"
    fixed_size = layer_state.size_mode == "Fixed"

    if not fixed_size:
        size_mask = isfinite(layer_state.layer[layer_state.size_attribute])
        mask = size_mask if mask is None else (mask & size_mask)
    if not fixed_color:
        color_mask = isfinite(layer_state.layer[layer_state.cmap_attribute])
        mask = color_mask if mask is None else (mask & color_mask)

    data = xyz_for_layer(viewer_state, layer_state,
                         preserve_aspect=viewer_state.native_aspect,
                         mask=mask,
                         scaled=True)
    data = data[:, [1, 2, 0]]
    factor = max((abs(b[1] - b[0]) for b in bounds))

    # We calculate this even if we aren't using fixed size as we might also use this for vectors
    radius = radius_for_scatter_layer(layer_state)
    if not fixed_size:
        # The specific size calculation is taken from the scatter layer artist
        size_data = ensure_numerical(layer_state.layer[layer_state.size_attribute][mask].ravel())
        size_data = clip(size_data, layer_state.size_vmin, layer_state.size_vmax)
        if layer_state.size_vmax == layer_state.size_vmin:
            sizes = sqrt(ones(size_data.shape) * 10)
        else:
            sizes = sqrt(((size_data - layer_state.size_vmin) /
                         (layer_state.size_vmax - layer_state.size_vmin)))
        sizes *= (layer_state.size_scaling / (2 * factor))
        sizes[isnan(sizes)] = 0.

    barr = bytearray()
    triangles = sphere_triangles(theta_resolution=theta_resolution, phi_resolution=phi_resolution)
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
    cmap_att = layer_state.cmap_attribute
    cmap_vals = layer_state.layer[cmap_att][mask]
    crange = layer_state.cmap_vmax - layer_state.cmap_vmin
    uri = f"layer_{unique_id()}.bin"
    for i, point in enumerate(data):

        prev_len = len(barr)
        r = radius if fixed_size else sizes[i]
        pts = sphere_points(center=point, radius=r,
                            theta_resolution=theta_resolution,
                            phi_resolution=phi_resolution)
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