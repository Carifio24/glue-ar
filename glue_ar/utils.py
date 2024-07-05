from glue.core.subset_group import GroupedSubset
from numpy import array, inf


def isomin_for_layer(viewer_state, layer):
    if isinstance(layer.layer, GroupedSubset):
        for viewer_layer in viewer_state.layers:
            if viewer_layer.layer is layer.layer.data:
                return viewer_layer.vmin

    return layer.vmin


def isomax_for_layer(viewer_state, layer):
    if isinstance(layer.layer, GroupedSubset):
        for viewer_layer in viewer_state.layers:
            if viewer_layer.layer is layer.layer.data:
                return viewer_layer.vmax

    return layer.vmax


def xyz_bounds(viewer_state):
    return [(viewer_state.x_min, viewer_state.x_max),
            (viewer_state.y_min, viewer_state.y_max),
            (viewer_state.z_min, viewer_state.z_max)]


def bounds_3d_from_layers(viewer_state, layer_states):
    mins = [inf, inf, inf]
    maxes = [-inf, -inf, -inf]
    atts = viewer_state.x_att, viewer_state.y_att, viewer_state.z_att
    for state in layer_states:
        data = state.layer.layer
        mins = [min(min(data[att]), m) for m, att in zip(mins, atts)]
        maxes = [max(max(data[att]), m) for m, att in zip(maxes, atts)]
    return [(lo, hi) for lo, hi in zip(mins, maxes)]


def slope_intercept_between(a, b):
    slope = (b[1] - a[1]) / (b[0] - a[0])
    intercept = b[1] - slope * b[0]
    return slope, intercept


# TODO: Make this better?
# glue-plotly has had to deal with similar issues,
# the utilities there are at least better than this
def layer_color(layer_state):
    layer_color = layer_state.color
    if layer_color == '0.35' or layer_color == '0.75':
        layer_color = '#808080'
    return layer_color


def bring_into_clip(data, bounds, preserve_aspect=True):
    if preserve_aspect:
        ranges = [abs(bds[1] - bds[0]) for bds in bounds]
        max_range = max(ranges)
        line_data = []
        for bds, rg in zip(bounds, ranges):
            frac = rg / max_range
            half_frac = frac / 2
            line_data.append(slope_intercept_between((bds[0], -half_frac), (bds[1], half_frac)))
    else:
        line_data = [slope_intercept_between((bds[0], -1), (bds[1], 1)) for bds in bounds]

    scaled = [[m * d + b for d in data[idx]] for idx, (m, b) in enumerate(line_data)]

    return scaled


def mask_for_bounds(viewer_state, layer_state, bounds):
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
def xyz_for_layer(viewer_state, layer_state,
                  scaled=False,
                  preserve_aspect=True,
                  mask=None):
    xs = layer_state.layer[viewer_state.x_att][mask]
    ys = layer_state.layer[viewer_state.y_att][mask]
    zs = layer_state.layer[viewer_state.z_att][mask]
    vals = [xs, ys, zs]

    if scaled:
        bounds = xyz_bounds(viewer_state)
        vals = bring_into_clip(vals, bounds, preserve_aspect=preserve_aspect)

    return array(list(zip(*vals)))


def hex_to_components(color):
    return [int(color[idx:idx+2], 16) for idx in range(1, len(color), 2)]
