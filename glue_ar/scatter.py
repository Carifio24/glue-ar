from numpy import array, clip, isnan, ones, sqrt
from numpy.linalg import norm
import pyvista as pv

from glue.utils import ensure_numerical
from glue_ar.utils import layer_color, mask_for_bounds, xyz_bounds, xyz_for_layer


# For the 3D scatter viewer
def scatter_layer_as_points(viewer_state, layer_state):
    xyz = xyz_for_layer(viewer_state, layer_state)
    return {
        "mesh": xyz,
        "color": layer_color(layer_state),
        "opacity": layer_state.alpha,
        "style": "points_gaussian",
        "point_size": 5 * layer_state.size,
        "render_points_as_spheres": True
    }


def scatter_layer_as_spheres(viewer_state, layer_state):
    data = xyz_for_layer(viewer_state, layer_state)
    return {
        "mesh": [pv.Sphere(center=p) for p in data]
    }


def scatter_layer_as_glyphs(viewer_state, layer_state, glyph):
    data = xyz_for_layer(viewer_state, layer_state, scaled=True)
    points = pv.PointSet(data)
    glyphs = points.glyph(geom=glyph, orient=False, scale=False)
    return {
        "mesh": glyphs,
        "color": layer_color(layer_state),
        "opacity": layer_state.alpha,
    }


_VECTOR_OFFSETS = {
    'tail': 0,
    'middle': -0.5,
    'tip': -1
}


def vector_meshes_for_layer(viewer_state, layer_state,
                            data, bounds,
                            tip_resolution=10,
                            shaft_resolution=10,
                            mask=None):
    atts = [layer_state.vx_attribute, layer_state.vy_attribute, layer_state.vz_attribute]
    tip_factor = 0.25 if layer_state.vector_arrowhead else 0
    vector_data = [layer_state.layer[att].ravel()[mask] for att in atts]
    if viewer_state.native_aspect:
        factor = max((abs(b[1] - b[0]) for b in bounds))
        vector_data = [[0.5 * t / factor for t in v] for v in vector_data]
    else:
        bound_factors = [abs(b[1] - b[0]) for b in bounds]
        vector_data = [[0.5 * t / b for t in v] for v, b in zip(vector_data, bound_factors)]
    vector_data = array(list(zip(*vector_data)))

    arrows = []
    offset = _VECTOR_OFFSETS[layer_state.vector_origin]
    for pt, v in zip(data, vector_data):
        adjusted_v = v * layer_state.vector_scaling
        length = norm(adjusted_v)
        tip_length = tip_factor * length
        adjusted_pt = [c + offset * vc for c, vc in zip(pt, adjusted_v)]
        arrow = pv.Arrow(
                start=adjusted_pt,
                direction=v,
                shaft_resolution=shaft_resolution,
                tip_resolution=tip_resolution,
                shaft_radius=0.002,
                tip_radius=0.01,
                scale=length,
                tip_length=tip_length)
        arrows.append(arrow)

    return arrows


def meshes_for_error_bars(viewer_state, layer_state, axis, data, bounds, mask=None):
    att = getattr(layer_state, f"{axis}err_attribute")
    err_values = layer_state.layer[att].ravel()[mask]
    index = ['x', 'y', 'z'].index(axis)
    axis_range = abs(bounds[index][1] - bounds[index][0])
    if viewer_state.native_aspect:
        max_range = max((abs(b[1] - b[0]) for b in bounds))
        factor = 1 / max_range
    else:
        factor = 1 / axis_range
    err_values *= factor
    lines = []
    for pt, err in zip(data, err_values):
        start = [c - err if idx == index else c for idx, c in enumerate(pt)]
        end = [c + err if idx == index else c for idx, c in enumerate(pt)]
        lines.append(pv.Line(start, end))

    return lines


# This function creates a multiblock mesh for a given scatter layer
# Everything is scaled into clip space for better usability with e.g. model-viewer
def scatter_layer_as_multiblock(viewer_state, layer_state,
                                size,
                                theta_resolution=8,
                                phi_resolution=8,
                                clip_to_bounds=True,
                                scaled=True):
    meshes = []
    bounds = xyz_bounds(viewer_state)
    if clip_to_bounds:
        mask = mask_for_bounds(viewer_state, layer_state, bounds)
    else:
        mask = None

    theta_resolution = int(theta_resolution)
    phi_resolution = int(phi_resolution)
    fixed_color = layer_state.color_mode == "Fixed"
    data = xyz_for_layer(viewer_state, layer_state,
                         preserve_aspect=viewer_state.native_aspect,
                         mask=mask,
                         scaled=scaled)
    viewer_area = size[0] * size[1]
    factor = 10
    if layer_state.size_mode == "Fixed":
        radius = factor * layer_state.size_scaling * sqrt(layer_state.size / viewer_area)
        spheres = [pv.Sphere(center=p, radius=radius,
                             phi_resolution=phi_resolution,
                             theta_resolution=theta_resolution) for p in data]
    else:
        # The specific size calculation is taken from the scatter layer artist
        size_data = ensure_numerical(layer_state.layer[layer_state.size_attribute][mask].ravel())
        size_data = clip(size_data, layer_state.size_vmin, layer_state.size_vmax)
        if layer_state.size_vmax == layer_state.size_vmin:
            sizes = sqrt(ones(size_data.shape) * 10)
        else:
            sizes = sqrt((20 * (size_data - layer_state.size_vmin) /
                     (layer_state.size_vmax - layer_state.size_vmin)))
        sizes *= (factor * layer_state.size_scaling / sqrt(viewer_area))
        sizes[isnan(sizes)] = 0.
        spheres = [pv.Sphere(center=p, radius=sqrt(s),
                             phi_resolution=phi_resolution,
                             theta_resolution=theta_resolution) for p, s in zip(data, sizes)]

    if not fixed_color:
        points_per_sphere = 2 + (phi_resolution - 2) * theta_resolution
        cmap_values = ensure_numerical(layer_state.layer[layer_state.cmap_attribute][mask].ravel())
        point_cmap_values = [y for x in cmap_values for y in (x,) * points_per_sphere]

    blocks = pv.MultiBlock(spheres)

    # Create the meshes for vectors, if necessary
    if layer_state.vector_visible:
        shaft_resolution = 10
        tip_resolution = 10
        arrows = vector_meshes_for_layer(viewer_state, layer_state,
                                         data, bounds,
                                         tip_resolution=tip_resolution,
                                         shaft_resolution=shaft_resolution,
                                         mask=mask)
        points_per_arrow = (4 * shaft_resolution) + tip_resolution + 1
        point_cmap_values.extend([y for x in cmap_values for y in (x,) * points_per_arrow])
        blocks.extend(arrows)

    geometry = blocks.extract_geometry()
    info = {
        "mesh": geometry,
        "opacity": layer_state.alpha
    }
    meshes.append(info)
    if fixed_color:
        info["color"] = layer_color(layer_state)
    else:
        geometry.point_data["colors"] = point_cmap_values
        cmap = layer_state.cmap.name  # This assumes that we're using a matplotlib colormap
        clim = [layer_state.cmap_vmin, layer_state.cmap_vmax]
        if clim[0] > clim[1]:
            clim = [clim[1], clim[0]]
            cmap = f"{cmap}_r"
        info["cmap"] = cmap
        info["clim"] = clim
        info["scalars"] = "colors"

    # Add error bars
    # We make these their own mesh because (for some reason) they disrupt the coloring of the
    # points and arrows if they're together in one MultiBlock
    # TODO: Why is this?
    if any((layer_state.xerr_visible, layer_state.yerr_visible, layer_state.zerr_visible)):
        bars = pv.MultiBlock()
        bars_info = {}
        bars_cmap_values = []
        for axis in ['x', 'y', 'z']:
            if getattr(layer_state, f"{axis}err_visible"):
                axis_bars = meshes_for_error_bars(viewer_state, layer_state,
                                             axis, data, bounds, mask=mask)
                bars.extend(axis_bars)
                if not fixed_color:
                    bars_cmap_values.extend([y for x in cmap_values for y in (x,) * 2])  # Each line has just two points

        bars_geometry = bars.extract_geometry()
        bars_info["mesh"] = bars_geometry
        if fixed_color:
            bars_info["color"] = layer_color(layer_state)
        else:
            bars_geometry.point_data["colors"] = bars_cmap_values
            bars_info["cmap"] = cmap
            bars_info["clim"] = clim
            bars_info["scalars"] = "colors"

        meshes.append(bars_info)

    return meshes
