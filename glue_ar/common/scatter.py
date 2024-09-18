from functools import partial
from numpy import clip, isfinite, isnan, ndarray, ones, sqrt
from typing import Callable, Dict, List, Optional, Tuple, Union

from glue.utils import ensure_numerical
from glue_vispy_viewers.scatter.layer_state import ScatterLayerState

from glue_ar.common.shapes import rectangular_prism_points, rectangular_prism_triangulation, \
                                  sphere_points, sphere_triangles
from glue_ar.utils import Bounds, NoneType, Viewer3DState, mask_for_bounds

try:
    from glue_jupyter.ipyvolume.scatter import Scatter3DLayerState
except (ModuleNotFoundError, ImportError):
    Scatter3DLayerState = NoneType

ScatterLayerState3D = Union[ScatterLayerState, Scatter3DLayerState]

Point = Tuple[float, float, float]
FullPointsGetter = Callable[[ScatterLayerState3D, Bounds, ndarray, Point, float], List[Point]]
PointsGetter = Callable[[Point, float], List[Point]]

VECTOR_OFFSETS = {
    'tail': 0.5,
    'middle': 0,
    'tip': -0.5,
}


def scatter_layer_mask(
        viewer_state: Viewer3DState,
        layer_state: ScatterLayerState3D,
        bounds: Bounds,
        clip_to_bounds: bool = True) -> ndarray:

    if clip_to_bounds:
        mask = mask_for_bounds(viewer_state, layer_state, bounds)
    else:
        mask = None

    vispy_layer_state = isinstance(layer_state, ScatterLayerState)
    fixed_size = layer_state.size_mode == "Fixed"
    cmap_mode_attr = "color_mode" if vispy_layer_state else "cmap_mode"
    fixed_color = getattr(layer_state, cmap_mode_attr, "Fixed") == "Fixed"
    size_attr = "size_attribute" if vispy_layer_state else "size_att"
    if not fixed_size:
        size_mask = isfinite(layer_state.layer[getattr(layer_state, size_attr)])
        mask = size_mask if mask is None else (mask & size_mask)
    cmap_attr = "cmap_attribute" if vispy_layer_state else "cmap_att"
    if not fixed_color:
        color_mask = isfinite(layer_state.layer[getattr(layer_state, cmap_attr)])
        mask = color_mask if mask is None else (mask & color_mask)

    return mask


def radius_for_scatter_layer(layer_state: ScatterLayerState3D) -> float:
    # This feels like a bit of a magic calculation, and it kind of is.
    # The motivation is as follows:
    # 30 is the largest size that we use in the vispy viewer - if the sizing of
    # a point would take it above 30, it's clipped to 30.
    # Looking at the vispy viewer, one can fit about 16 size-30 spheres
    # across one edge of the cube.
    # Hence we take the size of the vispy cube for scatter purposes to be 480
    return min(layer_state.size_scaling * layer_state.size, 30) / 480


def sizes_for_scatter_layer(layer_state: ScatterLayerState3D,
                            bounds: Bounds,
                            mask: ndarray) -> Optional[ndarray]:
    factor = max((abs(b[1] - b[0]) for b in bounds))
    vispy_layer_state = isinstance(layer_state, ScatterLayerState)
    if not vispy_layer_state:
        factor *= 2

    # We calculate this even if we aren't using fixed size as we might also use this for vectors
    fixed_size = layer_state.size_mode == "Fixed"
    if fixed_size:
        return None
    else:
        # The specific size calculation is taken from the scatter layer artist
        size_attr = "size_attribute" if vispy_layer_state else "size_att"
        size_data = ensure_numerical(layer_state.layer[getattr(layer_state, size_attr)][mask].ravel())
        size_data = clip(size_data, layer_state.size_vmin, layer_state.size_vmax)
        if layer_state.size_vmax == layer_state.size_vmin:
            sizes = sqrt(ones(size_data.shape) * 10)
        else:
            sizes = sqrt(((size_data - layer_state.size_vmin) /
                         (layer_state.size_vmax - layer_state.size_vmin)))
        sizes *= (layer_state.size_scaling / (2 * factor))
        sizes[isnan(sizes)] = 0.

    return sizes


def sphere_points_getter(theta_resolution: int,
                         phi_resolution: int) -> PointsGetter:

    return partial(sphere_points, theta_resolution=theta_resolution, phi_resolution=phi_resolution)


def box_points_getter(center: Point, size: float) -> List[Point]:
    return rectangular_prism_points(center=center, sides=[size, size, size])


IPYVOLUME_TRIANGLE_GETTERS: Dict[str, Callable] = {
    "box": rectangular_prism_triangulation,
    "sphere": partial(sphere_triangles, theta_resolution=13, phi_resolution=13),
    "diamond": partial(sphere_triangles, theta_resolution=3, phi_resolution=3),
    "circle_2d": partial(sphere_triangles, theta_resolution=13, phi_resolution=13),
}

IPYVOLUME_POINTS_GETTERS: Dict[str, PointsGetter] = {
    "box": box_points_getter,
    "sphere": sphere_points_getter(theta_resolution=13, phi_resolution=13),
    "diamond": sphere_points_getter(theta_resolution=3, phi_resolution=3),
    "circle_2d": sphere_points_getter(theta_resolution=13, phi_resolution=13),
}
