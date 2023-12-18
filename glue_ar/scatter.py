from math import cos, sin

import astropy.units as u 
from numpy import array

import pyvista as pv
from glue_ar.utils import layer_color, xyz_for_layer


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

def multiblock_for_layer(xyz_data, layer_state,
                         theta_resolution=8,
                         phi_resolution=8):
    spheres = [pv.Sphere(center=p, radius=layer_state.size_scaling * layer_state.size / 600, phi_resolution=phi_resolution, theta_resolution=theta_resolution) for p in xyz_data]
    blocks = pv.MultiBlock(spheres)
    geometry = blocks.extract_geometry()
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
        info["cmap"] = layer_state.cmap.name  # This assumes that we're using a matplotlib colormap
        info["clim"] = [layer_state.cmap_vmin, layer_state.cmap_vmax]
        info["scalars"] = "colors"
    return info


def scatter_layer_as_multiblock(viewer_state, layer_state, **kwargs):
    data = xyz_for_layer(layer_state, viewer_state, scaled=True)
    return multiblock_for_layer(data, layer_state, **kwargs)


def wwt_table_layer_as_multiblock(viewer_state, layer_state, **kwargs):
    from glue_wwt.viewer.viewer_state import MODES_3D
    if viewer_state.mode not in MODES_3D:
        raise ValueError("Viewer must be in a 3D mode to export")

    ra = layer_state.layer[viewer_state.lon_att]
    dec = layer_state.layer[viewer_state.lat_att]
    if viewer_state.alt_att is not None:
        alt = layer_state.layer[viewer_state.alt_att]
    else:
        alt = [1 for _ in range(len(ra))]

    factor = viewer_state.alt_unit.to(u.m).value
    alt = [factor * a for a in alt]

    # TODO: We need to do some more correcting here, based on alt_type

    x = [a * cos(d) * cos(r) for r, d, a in zip(ra, dec, alt)]
    y = [a * cos(d) * sin(r) for r, d, a in zip(ra, dec, alt)]
    z = [a * sin(d) for d, a in zip(dec, alt)]
    
    return array(list(zip(x, y, z)))

