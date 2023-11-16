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


def scatter_layer_as_multiblock(viewer_state, layer_state):
    data = xyz_for_layer(viewer_state, layer_state, scaled=True)
    spheres = [pv.Sphere(center=p, radius=0.01, phi_resolution=8, theta_resolution=8) for p in data]
    blocks = pv.MultiBlock(spheres)
    return {
        "data": blocks.extract_geometry(),
        "color": layer_color(layer_state),
        "opacity": layer_state.alpha
    }
