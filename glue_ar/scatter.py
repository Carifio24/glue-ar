from math import floor

import pyvista as pv
from glue_ar.utils import layer_color, xyz_bounds, xyz_for_layer


# For the 3D scatter viewer
def scatter_layer_as_points(viewer_state, layer_state):
    xyz = xyz_for_layer(viewer_state, layer_state)
    return {
        "data": xyz,
        "color": layer_color(layer_state),
        "opacity": layer_state.alpha,
        "style": "points_gaussian",
        "point_size": 5 * layer_state.size,
        "render_points_as_spheres": True
    }


def scatter_layer_as_spheres(viewer_state, layer_state):
    data = xyz_for_layer(viewer_state, layer_state)
    return {
        "data": [pv.Sphere(center=p) for p in data]
    }


def scatter_layer_as_glyphs(viewer_state, layer_state, glyph):
    data = xyz_for_layer(viewer_state, layer_state, scaled=True)
    points = pv.PointSet(data)
    glyphs = points.glyph(geom=glyph, orient=False, scale=False)
    return {
        "data": glyphs,
        "color": layer_color(layer_state),
        "opacity": layer_state.alpha,
    }


def rescale_spheres(mesh, viewer_state, data):
    bounds = xyz_bounds(viewer_state)
    factors = [abs(b[1] - b[0]) for b in bounds]
    n_spheres = len(data)
    points_per_sphere = len(mesh.points) / n_spheres
    for index, point in enumerate(mesh.points):
        sphere = floor(index / points_per_sphere)
        data_point = data[sphere]
        ds = point - data_point
        mesh.points[index] = [c + d / f for c, d, f in zip(data_point, ds, factors)]


def scatter_layer_as_multiblock(viewer_state, layer_state,
                                theta_resolution=8,
                                phi_resolution=8,
                                scaled=True):
    data = xyz_for_layer(viewer_state, layer_state, scaled=scaled, preserve_aspect=viewer_state.native_aspect)
    bounds = xyz_bounds(viewer_state)
    factor = max((abs(b[1] - b[0]) for b in bounds))
    radius = layer_state.size_scaling * layer_state.size
    if viewer_state.native_aspect:
        radius /= factor
    spheres = [pv.Sphere(center=p, radius=radius, phi_resolution=phi_resolution, theta_resolution=theta_resolution) for p in data]
    blocks = pv.MultiBlock(spheres)
    geometry = blocks.extract_geometry()

    if not viewer_state.native_aspect:
        rescale_spheres(geometry, viewer_state, data)

    info = {
        "data": geometry,
        "opacity": layer_state.alpha
    }
    if layer_state.color_mode == "Fixed":
        info["color"] = layer_color(layer_state)
    else:
        # sphere_cells = 2 * (phi_resolution - 2) * theta_resolution  # The number of cells on each sphere
        sphere_points = 2 + (phi_resolution - 2) * theta_resolution  # The number of points on each sphere
        cmap_values = layer_state.layer[layer_state.cmap_attribute]
        # cell_cmap_values = [y for x in cmap_values for y in (x,) * sphere_cells]
        point_cmap_values = [y for x in cmap_values for y in (x,) * sphere_points]
        # geometry.cell_data["colors"] = cell_cmap_values
        geometry.point_data["colors"] = point_cmap_values
        cmap = layer_state.cmap.name  # This assumes that we're using a matplotlib colormap
        clim = [layer_state.cmap_vmin, layer_state.cmap_vmax]
        if clim[0] > clim[1]:
            clim = [clim[1], clim[0]]
            cmap = f"{cmap}_r"
        info["cmap"] = cmap
        info["clim"] = clim
        info["scalars"] = "colors"

    return info
