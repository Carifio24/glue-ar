from collections import defaultdict
from gltflib import AccessorType, BufferTarget, ComponentType, PrimitiveMode
from glue.utils.array import ensure_numerical
from glue_vispy_viewers.common.viewer_state import Vispy3DViewerState
from glue_vispy_viewers.scatter.layer_state import ScatterLayerState
from numpy import ndarray
from numpy.linalg import norm

from typing import List, Literal, Optional, Tuple


from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.scatter_export_options import ARIpyvolumeScatterExportOptions, ARVispyScatterExportOptions
from glue_ar.common.shapes import cone_triangles, cone_points, cylinder_points, cylinder_triangles, \
                                  normalize, rectangular_prism_triangulation, sphere_triangles
from glue_ar.gltf_utils import add_points_to_bytearray, add_triangles_to_bytearray, index_export_option, \
                               index_mins, index_maxes
from glue_ar.utils import Viewer3DState, export_label_for_layer, iterable_has_nan, hex_to_components, \
                          layer_color, offset_triangles, unique_id, xyz_bounds, xyz_for_layer, Bounds
from glue_ar.common.gltf_builder import GLTFBuilder
from glue_ar.common.scatter import Scatter3DLayerState, ScatterLayerState3D, \
                                   PointsGetter, box_points_getter, IPYVOLUME_POINTS_GETTERS, \
                                   IPYVOLUME_TRIANGLE_GETTERS, VECTOR_OFFSETS, clip_error_data, clip_vector_data, \
                                   radius_for_scatter_layer, scatter_layer_mask, sizes_for_scatter_layer, \
                                   sphere_points_getter, NoneType

try:
    from glue_jupyter.common.state3d import ViewerState3D
except ImportError:
    ViewerState3D = NoneType


def add_vectors_gltf(builder: GLTFBuilder,
                     viewer_state: Viewer3DState,
                     layer_state: ScatterLayerState3D,
                     layer_id: str,
                     data: ndarray,
                     bounds: Bounds,
                     tip_height: float,
                     shaft_radius: float,
                     tip_radius: float,
                     tip_resolution: int = 6,
                     shaft_resolution: int = 6,
                     materials: Optional[dict[int, int]] = None,
                     mask: Optional[ndarray] = None):

    vector_data = clip_vector_data(viewer_state, layer_state, bounds, mask)
    offset = VECTOR_OFFSETS[layer_state.vector_origin]
    if layer_state.vector_origin == "tip":
        offset += tip_height

    barr = bytearray()
    triangles = cylinder_triangles(theta_resolution=shaft_resolution)
    triangle_count = len(triangles)
    max_index = max(idx for tri in triangles for idx in tri)
    add_triangles_to_bytearray(barr, triangles)

    index_format = index_export_option(max_index)
    if layer_state.vector_arrowhead:
        tip_triangles = cone_triangles(theta_resolution=tip_resolution, start_index=max_index + 1)
        add_triangles_to_bytearray(barr, tip_triangles, export_option=index_format)
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
        component_type=index_format.component_type,
        count=triangle_count*3,
        type=AccessorType.SCALAR,
        mins=[0],
        maxes=[max_index],
    )
    triangles_accessor = builder.accessor_count - 1

    point_mins = None
    point_maxes = None
    vispy_layer_state = isinstance(layer_state, ScatterLayerState)
    color_mode_attr = "color_mode" if vispy_layer_state else "cmap_mode"
    fixed_color = getattr(layer_state, color_mode_attr, "Fixed") == "Fixed"

    if not fixed_color:
        cmap_attr = "cmap_attribute" if vispy_layer_state else "cmap_att"
        cmap_att = getattr(layer_state, cmap_attr)
        cmap_vals = ensure_numerical(layer_state.layer[cmap_att][mask])
        crange = layer_state.cmap_vmax - layer_state.cmap_vmin

    for i, (pt, v) in enumerate(zip(data, vector_data)):
        if iterable_has_nan(v):
            continue

        if fixed_color:
            material_index = builder.material_count - 1
        else:
            cval = cmap_vals[i]
            normalized = max(min((cval - layer_state.cmap_vmin) / crange, 1), 0)
            cindex = int(normalized * 255)
            material_index = materials[cindex] if materials else builder.material_count - 1

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
            layer_id=layer_id,
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
                        layer_id: str,
                        axis: Literal["x", "y", "z"],
                        data: ndarray,
                        bounds: Bounds,
                        materials: Optional[dict[int, int]] = None,
                        mask: Optional[ndarray] = None):
    err_values = clip_error_data(viewer_state, layer_state, bounds, axis, mask)

    vispy_layer_state = isinstance(layer_state, ScatterLayerState)
    color_mode_attr = "color_mode" if vispy_layer_state else "cmap_mode"
    fixed_color = getattr(layer_state, color_mode_attr, "Fixed") == "Fixed"
    if not fixed_color:
        cmap_attr = "cmap_attribute" if vispy_layer_state else "cmap_att"
        cmap_att = getattr(layer_state, cmap_attr)
        cmap_vals = ensure_numerical(layer_state.layer[cmap_att][mask])
        crange = layer_state.cmap_vmax - layer_state.cmap_vmin

    # NB: This ordering is intentional to account for glTF coordinate system
    gltf_index = ['y', 'z', 'x'].index(axis)

    barr = bytearray()
    errors_bin = f"errors_{unique_id()}.bin"
    segments_by_material = defaultdict(list)
    material_index = builder.material_count - 1
    for i, (pt, err) in enumerate(zip(data, err_values)):

        if not fixed_color:
            cval = cmap_vals[i]
            normalized = max(min((cval - layer_state.cmap_vmin) / crange, 1), 0)
            cindex = int(normalized * 255)
            material_index = materials[cindex] if materials else builder.material_count - 1

        start = [c - err if idx == gltf_index else c for idx, c in enumerate(pt)]
        end = [c + err if idx == gltf_index else c for idx, c in enumerate(pt)]
        line_points = (start, end)
        segments_by_material[material_index].extend(line_points)

    for material, segments in segments_by_material.items():
        bv_start = len(barr)
        add_points_to_bytearray(barr, segments)
        pt_mins = index_mins(segments)
        pt_maxes = index_maxes(segments)
        bv_len = len(barr) - bv_start

        builder.add_buffer_view(
            buffer=builder.buffer_count,
            byte_length=bv_len,
            byte_offset=bv_start,
            target=BufferTarget.ARRAY_BUFFER,
        )
        builder.add_accessor(
            buffer_view=builder.buffer_view_count-1,
            component_type=ComponentType.FLOAT,
            count=len(segments),
            type=AccessorType.VEC3,
            mins=pt_mins,
            maxes=pt_maxes,
        )
        builder.add_mesh(
            layer_id=layer_id,
            position_accessor=builder.accessor_count-1,
            material=material,
            mode=PrimitiveMode.LINES,
        )

    builder.add_buffer(byte_length=len(barr), uri=errors_bin)
    builder.add_file_resource(errors_bin, data=barr)


def add_scatter_layer_gltf(builder: GLTFBuilder,
                           viewer_state: Viewer3DState,
                           layer_state: ScatterLayerState3D,
                           points_getter: PointsGetter,
                           triangles: List[Tuple[int, int, int]],
                           bounds: Bounds,
                           clip_to_bounds: bool = True,
                           points_per_mesh: Optional[int] = None):
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
    if len(data) == 0:
        return

    data = data[:, [1, 2, 0]]

    buffer = builder.buffer_count
    cmap = layer_state.cmap
    cmap_attr = "cmap_attribute" if vispy_layer_state else "cmap_att"
    cmap_att = getattr(layer_state, cmap_attr)
    cmap_vals = ensure_numerical(layer_state.layer[cmap_att][mask])
    crange = layer_state.cmap_vmax - layer_state.cmap_vmin
    uri = f"layer_{unique_id()}.bin"

    sizes = sizes_for_scatter_layer(layer_state, bounds, mask)

    barr = bytearray()
    n_points = len(data)

    # If points per mesh is not specified,
    # we don't do any mesh chunking.
    # Note that this will work for colormapping as well
    if points_per_mesh is None:
        points_per_mesh = n_points

    layer_id = export_label_for_layer(layer_state)

    if fixed_color:
        points = []
        tris = []

        color = layer_color(layer_state)
        color_components = hex_to_components(color)
        builder.add_material(color=color_components, opacity=layer_state.alpha)

        for i, point in enumerate(data):
            size = radius if fixed_size else sizes[i]
            pts = points_getter(point, size)
            points.append(pts)

        # If n_points is less than our designated chunk size, we only want
        # to make triangles for that many points (and put everything in one mesh).
        # This is both more space-efficient and necessary to be glTF spec-compliant
        triangle_offset = 0
        pts_count = len(points_getter((0, 0, 0), 1))
        for _ in range(min(points_per_mesh, n_points)):
            pt_triangles = offset_triangles(triangles, triangle_offset)
            triangle_offset += pts_count
            tris.append(pt_triangles)

        mesh_triangles = [tri for sphere in tris for tri in sphere]
        max_triangle_index = max(idx for tri in mesh_triangles for idx in tri)
        index_format = index_export_option(max_triangle_index)
        triangles_start = len(barr)
        add_triangles_to_bytearray(barr, mesh_triangles, export_option=index_format)
        triangles_len = len(barr)
        builder.add_buffer_view(
            buffer=buffer,
            byte_length=triangles_len-triangles_start,
            byte_offset=triangles_start,
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

        start = 0
        triangles_accessor = builder.accessor_count - 1
        while start < n_points:
            mesh_points = [pt for pts in points[start:start+points_per_mesh] for pt in pts]
            barr_offset = len(barr)
            add_points_to_bytearray(barr, mesh_points)
            point_mins = index_mins(mesh_points)
            point_maxes = index_maxes(mesh_points)

            builder.add_buffer_view(
                buffer=buffer,
                byte_length=len(barr)-barr_offset,
                byte_offset=barr_offset,
                target=BufferTarget.ARRAY_BUFFER,
            )
            builder.add_accessor(
                buffer_view=builder.buffer_view_count-1,
                component_type=ComponentType.FLOAT,
                count=len(mesh_points),
                type=AccessorType.VEC3,
                mins=point_mins,
                maxes=point_maxes,
            )
            points_accessor = builder.accessor_count - 1

            # This should only happen on the final iteration
            # or not at all, if points_per_mesh is a divisor of count
            # But in this case we do need a separate accessor as the
            # byte length is different.
            # Note that if we're on the first chunk (start == 0)
            # there's no need to do this - we can use the buffer view that we just created
            count = n_points - start
            if start != 0 and count < points_per_mesh:
                triangles_count = len(tris)
                byte_length = count * triangles_len // triangles_count
                mesh_triangles = [tri for sphere in tris[:count] for tri in sphere]
                max_triangle_index = max(idx for tri in mesh_triangles for idx in tri)
                builder.add_buffer_view(
                    buffer=buffer,
                    byte_length=byte_length,
                    byte_offset=triangles_start,
                    target=BufferTarget.ELEMENT_ARRAY_BUFFER,
                )
                builder.add_accessor(
                    buffer_view=builder.buffer_view_count-1,
                    component_type=index_format.component_type,
                    count=len(triangles)*3*count,
                    type=AccessorType.SCALAR,
                    mins=[0],
                    maxes=[max_triangle_index],
                )
                triangles_accessor = builder.accessor_count - 1

            builder.add_mesh(
                layer_id=layer_id,
                position_accessor=points_accessor,
                indices_accessor=triangles_accessor,
                material=builder.material_count-1,
            )
            start += points_per_mesh

    else:
        points_by_color = defaultdict(list)
        color_materials = defaultdict(int)

        for i, point in enumerate(data):
            cval = cmap_vals[i]
            normalized = max(min((cval - layer_state.cmap_vmin) / crange, 1), 0)
            cindex = int(normalized * 255)
            color = cmap(cindex)

            material_index = color_materials.get(cindex, None)
            if material_index is None:
                builder.add_material(color, layer_state.alpha)
                material_index = builder.material_count - 1
                color_materials[cindex] = material_index

            size = radius if fixed_size else sizes[i]
            pts = points_getter(point, size)
            points_by_color[cindex].append(pts)

        for cindex, points in points_by_color.items():

            # If the maximum number of points in any one color is less than our designated chunk size,
            # we only want to make triangles for that many points (and put everything in one mesh).
            # This is both more space-efficient and necessary to be glTF spec-compliant
            triangle_offset = 0
            tris = []
            pts_count = len(points_getter((0, 0, 0), 1))
            for _ in range(min(points_per_mesh, len(points))):
                pt_triangles = offset_triangles(triangles, triangle_offset)
                triangle_offset += pts_count
                tris.append(pt_triangles)

            triangles_count = len(tris)
            mesh_points = [pt for pts in points for pt in pts]
            mesh_triangles = [tri for sphere in tris for tri in sphere]
            max_triangle_index = max(idx for tri in mesh_triangles for idx in tri)
            index_format = index_export_option(max_triangle_index)
            triangles_start = len(barr)
            add_triangles_to_bytearray(barr, mesh_triangles, export_option=index_format)
            triangles_len = len(barr)

            builder.add_buffer_view(
                buffer=buffer,
                byte_length=triangles_len-triangles_start,
                byte_offset=triangles_start,
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

            start = 0
            triangles_accessor = builder.accessor_count - 1
            n_points = len(points)
            while start < n_points:
                mesh_points = [pt for pts in points[start:start+points_per_mesh] for pt in pts]
                barr_offset = len(barr)
                add_points_to_bytearray(barr, mesh_points)
                point_mins = index_mins(mesh_points)
                point_maxes = index_maxes(mesh_points)

                builder.add_buffer_view(
                    buffer=buffer,
                    byte_length=len(barr)-barr_offset,
                    byte_offset=barr_offset,
                    target=BufferTarget.ARRAY_BUFFER,
                )
                builder.add_accessor(
                    buffer_view=builder.buffer_view_count-1,
                    component_type=ComponentType.FLOAT,
                    count=len(mesh_points),
                    type=AccessorType.VEC3,
                    mins=point_mins,
                    maxes=point_maxes,
                )
                points_accessor = builder.accessor_count - 1

                # This should only happen on the final iteration
                # or not at all, if points_per_mesh is a divisor of count
                # But in this case we do need a separate accessor as the
                # byte length is different.
                # Note that if we're on the first chunk (start == 0)
                # there's no need to do this - we can use the buffer view that we just created
                count = n_points - start
                if start != 0 and count < points_per_mesh:
                    byte_length = count * triangles_len // triangles_count
                    mesh_triangles = [tri for sphere in tris[:count] for tri in sphere]
                    max_triangle_index = max(idx for tri in mesh_triangles for idx in tri)
                    builder.add_buffer_view(
                        buffer=buffer,
                        byte_length=byte_length,
                        byte_offset=triangles_start,
                        target=BufferTarget.ELEMENT_ARRAY_BUFFER,
                    )
                    builder.add_accessor(
                        buffer_view=builder.buffer_view_count-1,
                        component_type=index_format.component_type,
                        count=len(triangles)*3*count,
                        type=AccessorType.SCALAR,
                        mins=[0],
                        maxes=[max_triangle_index],
                    )
                    triangles_accessor = builder.accessor_count - 1

                material = color_materials[cindex]
                builder.add_mesh(
                    layer_id=layer_id,
                    position_accessor=points_accessor,
                    indices_accessor=triangles_accessor,
                    material=material,
                )
                start += points_per_mesh

    builder.add_buffer(byte_length=len(barr), uri=uri)
    builder.add_file_resource(uri, data=barr)

    materials = color_materials if not fixed_color else None
    for axis in ("x", "y", "z"):
        if getattr(layer_state, f"{axis}err_visible", False):
            add_error_bars_gltf(
                builder=builder,
                viewer_state=viewer_state,
                layer_state=layer_state,
                layer_id=layer_id,
                axis=axis,
                data=data,
                bounds=bounds,
                materials=materials,
                mask=mask,
            )

    if layer_state.vector_visible:
        shaft_radius = radius / 1.5
        tip_radius = shaft_radius * 4
        tip_height = 2 * tip_radius
        add_vectors_gltf(
            builder=builder,
            viewer_state=viewer_state,
            layer_state=layer_state,
                layer_id=layer_id,
            data=data,
            bounds=bounds,
            tip_height=tip_height,
            shaft_radius=shaft_radius,
            tip_radius=tip_radius,
            shaft_resolution=6,
            tip_resolution=6,
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

    resolution = int(options.resolution)
    theta_resolution = resolution
    phi_resolution = resolution
    triangles = sphere_triangles(theta_resolution=theta_resolution,
                                 phi_resolution=phi_resolution)

    points_getter = sphere_points_getter(theta_resolution=theta_resolution,
                                         phi_resolution=phi_resolution)
    log_ppm = int(options.log_points_per_mesh)
    if log_ppm == 7:
        ppm = None
    else:
        ppm = 10 ** log_ppm

    add_scatter_layer_gltf(builder=builder,
                           viewer_state=viewer_state,
                           layer_state=layer_state,
                           points_getter=points_getter,
                           triangles=triangles,
                           bounds=bounds,
                           clip_to_bounds=clip_to_bounds,
                           points_per_mesh=ppm)


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
    log_ppm = int(options.log_points_per_mesh)
    if log_ppm == 7:
        ppm = None
    else:
        ppm = 10 ** log_ppm

    add_scatter_layer_gltf(builder=builder,
                           viewer_state=viewer_state,
                           layer_state=layer_state,
                           points_getter=points_getter,
                           triangles=triangles,
                           bounds=bounds,
                           clip_to_bounds=clip_to_bounds,
                           points_per_mesh=ppm)
