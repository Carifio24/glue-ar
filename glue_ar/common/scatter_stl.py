from typing import List, Tuple

from glue_vispy_viewers.common.viewer_state import Vispy3DViewerState
from glue_vispy_viewers.scatter.layer_state import ScatterLayerState

from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.scatter import PointsGetter, ScatterLayerState3D, radius_for_scatter_layer, scatter_layer_mask, sizes_for_scatter_layer, sphere_points_getter
from glue_ar.common.scatter_export_options import ARVispyScatterExportOptions
from glue_ar.common.shapes import sphere_triangles
from glue_ar.common.stl_builder import STLBuilder
from glue_ar.utils import Bounds, Viewer3DState, xyz_bounds, xyz_for_layer


def add_scatter_layer_stl(builder: STLBuilder,
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

    sizes = sizes_for_scatter_layer(layer_state, bounds, mask)
    for i, point in enumerate(data):

        size = radius if fixed_size else sizes[i]
        pts = points_getter(point, size)
        builder.add_mesh(pts, triangles)


@ar_layer_export(ScatterLayerState, "Scatter", ARVispyScatterExportOptions, ("stl",))
def add_vispy_scatter_layer_stl(builder: STLBuilder,
                                viewer_state: Vispy3DViewerState,
                                layer_state: ScatterLayerState,
                                options: ARVispyScatterExportOptions,
                                bounds: Bounds,
                                clip_to_bounds: bool = True):

    triangles = sphere_triangles(theta_resolution=options.theta_resolution,
                                 phi_resolution=options.phi_resolution)

    points_getter = sphere_points_getter(theta_resolution=options.theta_resolution,
                                         phi_resolution=options.phi_resolution)

    add_scatter_layer_stl(builder=builder,
                         viewer_state=viewer_state,
                         layer_state=layer_state,
                         points_getter=points_getter,
                         triangles=triangles,
                         bounds=bounds,
                         clip_to_bounds=clip_to_bounds)
