from numbers import Number
from os.path import abspath, dirname, join
from uuid import uuid4
from typing import Iterator, Literal, overload, Iterable, List, Optional, Tuple, Union

from glue.core import BaseData
from glue.core.subset_group import GroupedSubset
from glue.viewers.common.state import ViewerState
from glue.viewers.common.viewer import LayerArtist, Viewer

from glue_vispy_viewers.common.layer_state import LayerState, VispyLayerState
from glue_vispy_viewers.volume.volume_viewer import VispyVolumeViewerMixin
from glue_vispy_viewers.volume.layer_state import VolumeLayerState
from glue_vispy_viewers.volume.viewer_state import Vispy3DViewerState, Vispy3DVolumeViewerState
from numpy import array, inf, isnan, ndarray

try:
    from glue_jupyter.common.state3d import ViewerState3D
except (ImportError, ModuleNotFoundError):
    ViewerState3D = Vispy3DViewerState

# Backwards compatibility for Python < 3.10
try:
    from types import NoneType  # noqa
except (ImportError, ModuleNotFoundError):
    NoneType = type(None)

PACKAGE_DIR = dirname(abspath(__file__))
AR_ICON = abspath(join(dirname(__file__), "ar.png"))
RESOURCES_DIR = join(PACKAGE_DIR, "resources")

Bounds = List[Tuple[float, float]]
BoundsWithResolution = List[Tuple[float, float, int]]

Viewer3DState = Union[Vispy3DViewerState, ViewerState3D]


def data_count(layers: Iterable[Union[LayerArtist, LayerState]]) -> int:
    """
    Count the number of unique Data objects (either directly or as parents of subsets)
    used in the set of layers
    """
    data = set(layer.layer if isinstance(layer.layer, BaseData) else layer.layer.data for layer in layers)
    return len(data)


def export_label_for_layer(layer: Union[LayerArtist, LayerState],
                           add_data_label: bool = True) -> str:
    if (not add_data_label) or isinstance(layer.layer, BaseData):
        return layer.layer.label
    else:
        data = layer.layer.data
        return f"{layer.layer.label} ({data.label})"


def layers_to_export(viewer: Viewer) -> List[LayerArtist]:
    return list(filter(lambda artist: artist.enabled and artist.visible, viewer.layers))


def isomin_for_layer(viewer_state: ViewerState, layer_state: VolumeLayerState) -> float:
    if isinstance(layer_state.layer, GroupedSubset):
        for viewer_layer in viewer_state.layers:
            if viewer_layer.layer is layer_state.layer.data:
                return viewer_layer.vmin

    return layer_state.vmin


def isomax_for_layer(viewer_state: ViewerState, layer_state: VolumeLayerState) -> float:
    if isinstance(layer_state.layer, GroupedSubset):
        for viewer_layer in viewer_state.layers:
            if viewer_layer.layer is layer_state.layer.data:
                return viewer_layer.vmax

    return layer_state.vmax


@overload
def xyz_bounds(viewer_state: Viewer3DState, with_resolution: Literal[False]) -> Bounds: ...


@overload
def xyz_bounds(viewer_state: Viewer3DState, with_resolution: Literal[True]) -> BoundsWithResolution: ...


def xyz_bounds(viewer_state: Viewer3DState, with_resolution: bool) -> Union[Bounds, BoundsWithResolution]:
    bounds: Bounds = [(viewer_state.x_min, viewer_state.x_max),
                      (viewer_state.y_min, viewer_state.y_max),
                      (viewer_state.z_min, viewer_state.z_max)]
    if with_resolution:
        resolution = get_resolution(viewer_state)
        return [(*b, resolution) for b in bounds]

    return bounds


@overload
def bounds_3d_from_layers(viewer_state: Viewer3DState,
                          layer_states: Iterable[VispyLayerState],
                          with_resolution: Literal[False]) -> Bounds: ...


@overload
def bounds_3d_from_layers(viewer_state: Viewer3DState,
                          layer_states: Iterable[VispyLayerState],
                          with_resolution: Literal[True]) -> BoundsWithResolution: ...


def bounds_3d_from_layers(viewer_state: Viewer3DState,
                          layer_states: Iterable[VispyLayerState],
                          with_resolution: bool) -> Union[Bounds, BoundsWithResolution]:
    mins = [inf, inf, inf]
    maxes = [-inf, -inf, -inf]
    atts = viewer_state.x_att, viewer_state.y_att, viewer_state.z_att
    for state in layer_states:
        data = state.layer.layer
        mins = [min(min(data[att]), m) for m, att in zip(mins, atts)]
        maxes = [max(max(data[att]), m) for m, att in zip(maxes, atts)]
    bounds = [(lo, hi) for lo, hi in zip(mins, maxes)]
    if with_resolution:
        resolution = get_resolution(viewer_state)
        return [(*b, resolution) for b in bounds]

    return bounds


def slope_intercept_between(a: List[float], b: List[float]) -> Tuple[float, float]:
    slope = (b[1] - a[1]) / (b[0] - a[0])
    intercept = b[1] - slope * b[0]
    return slope, intercept


# TODO: Make this better?
# glue-plotly has had to deal with similar issues,
# the utilities there are at least better than this
def layer_color(layer_state: LayerState) -> str:
    layer_color = layer_state.color
    if layer_color == '0.35' or layer_color == '0.75':
        layer_color = '#808080'
    return layer_color


def bring_into_clip(data, bounds: Union[Bounds, BoundsWithResolution], preserve_aspect: bool = True):
    if preserve_aspect:
        ranges = [abs(bds[1] - bds[0]) for bds in bounds]
        max_range = max(ranges)
        line_data = []
        for bds, rg in zip(bounds, ranges):
            frac = rg / max_range
            line_data.append(slope_intercept_between([bds[0], -frac], [bds[1], frac]))
    else:
        line_data = [slope_intercept_between([bds[0], -1], [bds[1], 1]) for bds in bounds]

    scaled = [[m * d + b for d in data[idx]] for idx, (m, b) in enumerate(line_data)]

    return scaled


def mask_for_bounds(viewer_state: Viewer3DState,
                    layer_state: LayerState,
                    bounds: Union[Bounds, BoundsWithResolution]):
    data = layer_state.layer
    bounds = [(min(b), max(b)) for b in bounds]
    return (data[viewer_state.x_att] >= bounds[0][0]) & \
           (data[viewer_state.x_att] <= bounds[0][1]) & \
           (data[viewer_state.y_att] >= bounds[1][0]) & \
           (data[viewer_state.y_att] <= bounds[1][1]) & \
           (data[viewer_state.z_att] >= bounds[2][0]) & \
           (data[viewer_state.z_att] <= bounds[2][1])


# TODO: Worry about efficiency later
# and just generally make this better
def xyz_for_layer(viewer_state: Viewer3DState,
                  layer_state: LayerState,
                  scaled: bool = False,
                  preserve_aspect: bool = True,
                  mask: Optional[ndarray] = None) -> ndarray:
    xs = layer_state.layer[viewer_state.x_att][mask]
    ys = layer_state.layer[viewer_state.y_att][mask]
    zs = layer_state.layer[viewer_state.z_att][mask]
    vals = [xs, ys, zs]

    if scaled:
        bounds = xyz_bounds(viewer_state, with_resolution=False)
        vals = bring_into_clip(vals, bounds, preserve_aspect=preserve_aspect)

    return array(list(zip(*vals)))


def hex_to_components(color: str) -> List[int]:
    return [int(color[idx:idx+2], 16) for idx in range(1, len(color), 2)]


def unique_id() -> str:
    return uuid4().hex


def alpha_composite(over: List[float], under: List[float]) -> List[float]:
    alpha_o = over[3] if len(over) == 4 else over[2]
    alpha_u = under[3] if len(under) == 4 else under[2]
    rgb_o = over[:3]
    rgb_u = under[:3]
    alpha_new = alpha_o + alpha_u * (1 - alpha_o)
    rgba_new = [
        (co * alpha_o + cu * alpha_u * (1 - alpha_o)) / alpha_new
        for co, cu in zip(rgb_o, rgb_u)
    ]
    rgba_new.append(alpha_new)
    return rgba_new


def data_for_layer(layer_or_state: Union[LayerArtist, LayerState]) -> BaseData:
    if isinstance(layer_or_state.layer, BaseData):
        return layer_or_state.layer
    else:
        return layer_or_state.layer.data


def frb_for_layer(viewer_state: ViewerState,
                  layer_or_state: Union[LayerArtist, LayerState],
                  bounds: BoundsWithResolution) -> ndarray:

    bounds = list(reversed(bounds))
    data = data_for_layer(layer_or_state)
    layer_state = layer_or_state if isinstance(layer_or_state, LayerState) else layer_or_state.state
    is_data_layer = data is layer_or_state.layer
    target_data = getattr(viewer_state, 'reference_data', data)
    data_frb = data.compute_fixed_resolution_buffer(
        target_data=target_data,
        bounds=bounds,
        target_cid=layer_state.attribute
    )

    if is_data_layer:
        return data_frb
    else:
        subcube = data.compute_fixed_resolution_buffer(
            target_data=target_data,
            bounds=bounds,
            subset_state=layer_state.layer.subset_state
        )
        return subcube * data_frb


def ndarray_has_nan(arr: ndarray) -> bool:
    return bool(isnan(arr).any())


def iterable_has_nan(arr: Iterable[float]) -> bool:
    return any(isnan(x) for x in arr)


def iterator_count(iter: Iterator) -> int:
    """
    Returns the size of an iterator.
    Note that this consumes the iterator.
    """
    return sum(1 for _ in iter)


def is_volume_viewer(viewer: Viewer) -> bool:
    if isinstance(viewer, VispyVolumeViewerMixin):
        return True
    try:
        from glue_jupyter.ipyvolume.volume import IpyvolumeVolumeView
        if isinstance(viewer, IpyvolumeVolumeView):
            return True
    except (ImportError, ModuleNotFoundError):
        pass

    return False


def get_resolution(viewer_state: Viewer3DState) -> int:
    if isinstance(viewer_state, Vispy3DVolumeViewerState):
        return viewer_state.resolution

    try:
        from glue_jupyter.common.state3d import VolumeViewerState
        if isinstance(viewer_state, VolumeViewerState):
            return max((getattr(state, 'max_resolution', 256) for state in viewer_state.layers), default=256)
    except (ImportError, ModuleNotFoundError):
        pass

    return 256


# TODO: What is the right typing here?
def clamp(value: Number, minimum: Number, maximum: Number) -> Number:
    return min(max(value, minimum), maximum)


def clamped_opacity(opacity: float) -> float:
    return clamp(opacity, 0, 1)


def binned_opacity(raw_opacity: float, resolution: float) -> float:
    return clamped_opacity(round(raw_opacity / resolution) * resolution)
