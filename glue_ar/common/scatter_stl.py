from typing import List, Tuple

from glue_vispy_viewers.scatter.layer_state import ScatterLayerState

from glue_ar.common.scatter import PointsGetter, ScatterLayerState3D, radius_for_scatter_layer, scatter_layer_mask
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

