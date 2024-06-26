from itertools import product
import numpy as np
from numpy import clip, full, invert, isfinite, isnan
import pyvista as pv
from scipy.ndimage import gaussian_filter

from glue.core.subset_group import GroupedSubset
from glue_ar.utils import hex_color, isomin_for_layer, isomax_for_layer, layer_color, rgba_components


def volume_layer_as_multiblock(viewer_state, layer_state, bounds,
                               use_gaussian_filter=False,
                               precomputed_frbs=None,
                               **kwargs):

    layer_content = layer_state.layer
    parent = layer_content.data if isinstance(layer_content, GroupedSubset) else layer_content

    parent_label = parent.label
    if precomputed_frbs is not None and parent_label in precomputed_frbs:
        data = precomputed_frbs[parent_label]
    else:
        data = parent.compute_fixed_resolution_buffer(
            target_data=viewer_state.reference_data,
            bounds=bounds,
            target_cid=layer_state.attribute)
        if precomputed_frbs is not None:
            precomputed_frbs[parent_label] = data
        data[~isfinite(data)] = 0

    if isinstance(layer_state.layer, GroupedSubset):
        subcube = parent.compute_fixed_resolution_buffer(
            target_data=viewer_state.reference_data,
            bounds=bounds,
            subset_state=layer_state.layer.subset_state
        )
        data = subcube * data
        data[isnan(data)] = 0.

    if use_gaussian_filter:
        data = gaussian_filter(data, 1)

    isomin = isomin_for_layer(viewer_state, layer_state)
    isomax = isomax_for_layer(viewer_state, layer_state)

    # Conventions between pyvista and glue data storage
    data = data.transpose(2, 1, 0)

    size_x = (viewer_state.x_max - viewer_state.x_min) / viewer_state.resolution
    size_y = (viewer_state.y_max - viewer_state.y_min) / viewer_state.resolution
    size_z = (viewer_state.z_max - viewer_state.z_min) / viewer_state.resolution

    cubes = []
    for i, j, k in product(range(viewer_state.resolution), repeat=3):
        center_x = (i + 0.5) * size_x
        center_y = (i + 0.5) * size_y
        center_z = (i + 0.5) * size_z
        cubes.append(pv.Cube(center=(center_x, center_y, center_z),
                             x_length=size_x, y_length=size_y, z_length=size_z))

    blocks = pv.MultiBlock(cubes)

    return [{
        "mesh": blocks.extract_geometry(),
        "opacity": layer_state.alpha
    }]

        


# Trying to export each layer individually, rather than doing all the meshes
# as a global operation on the viewer.
# The main difference here is that we aren't excising the subset points from
# the parent mesh as Luca's plugin did.
# But glue isn't going to be doing that, and if we have opacity then we should(?)
# get the same effect as in glue in the exported file
def meshes_for_volume_layer(viewer_state, layer_state, bounds,
                            use_gaussian_filter=False,
                            precomputed_frbs=None,
                            **kwargs):

    layer_content = layer_state.layer
    parent = layer_content.data if isinstance(layer_content, GroupedSubset) else layer_content

    parent_label = parent.label
    if precomputed_frbs is not None and parent_label in precomputed_frbs:
        data = precomputed_frbs[parent_label]
    else:
        data = parent.compute_fixed_resolution_buffer(
            target_data=viewer_state.reference_data,
            bounds=bounds,
            target_cid=layer_state.attribute)
        if precomputed_frbs is not None:
            precomputed_frbs[parent_label] = data
        data[~isfinite(data)] = 0

    if isinstance(layer_state.layer, GroupedSubset):
        subcube = parent.compute_fixed_resolution_buffer(
            target_data=viewer_state.reference_data,
            bounds=bounds,
            subset_state=layer_state.layer.subset_state
        )
        data = subcube * data
        data[isnan(data)] = 0.

    if use_gaussian_filter:
        data = gaussian_filter(data, 1)

    isomin = isomin_for_layer(viewer_state, layer_state)
    isomax = isomax_for_layer(viewer_state, layer_state)

    # Conventions between pyvista and glue data storage
    data = data.transpose(2, 1, 0)

    grid = pv.ImageData()
    grid.dimensions = (viewer_state.resolution,) * 3
    grid.origin = (0, 0, 0)

    # Comment from Luca: # I think the voxel spacing will always be 1,
    # because of how glue downsamples to a fixed resolution grid. But don't hold me to this!
    #
    # However, we're not using that idea anymore - the spacing entries can be floats,
    # so we just calculate them based on the axis ranges and the resolution

    ranges = (
        viewer_state.x_max - viewer_state.x_min,
        viewer_state.y_max - viewer_state.y_min,
        viewer_state.z_max - viewer_state.z_min
    )
    max_range = max(ranges)

    if viewer_state.native_aspect:
        grid.spacing = tuple(r / viewer_state.resolution for r in ranges)
    else:
        grid.spacing = (1 / (max_range * viewer_state.resolution),) * 3
    values = data.flatten(order="F")
    opacities = values - isomin
    opacities *= layer_state.alpha / (isomax - isomin)
    clip(opacities, 0, 1, out=opacities)
    grid.point_data["values"] = values
    # grid.point_data["opacities"] = opacities

    color = layer_color(layer_state)
    # colors = full(values.shape, 0.5)
    # We need a "colormap" to map the scalars to
    r, g, b, _ = rgba_components(color)
    cmap_size = 32
    cmap = [hex_color(r, g, b, 256 * i / cmap_size) for i in range(cmap_size)]
    print(cmap)
    # grid.point_data["colors"] = colors

    return [{
        "mesh": grid,
        "color": color,
        # "opacity": "opacities",
        "scalars": "values",
        "cmap": cmap,
        "clim": [isomin, isomax],
        "nan_color": "#ffffff",
        "nan_opacity": 0.0,
    }]


def isosurface_meshes_for_layer(viewer_state, layer_state, bounds,
                                use_gaussian_filter=False,
                                smoothing_iterations=0,
                                isosurface_count=5,
                                precomputed_frbs=None):

    mesh_data = meshes_for_volume_layer(viewer_state, layer_state, bounds,
                                        use_gaussian_filter=use_gaussian_filter,
                                        precomputed_frbs=precomputed_frbs)

    isomin = isomin_for_layer(viewer_state, layer_state)
    isomax = isomax_for_layer(viewer_state, layer_state)
    grid = mesh_data.pop("mesh")

    isosurfaces = np.linspace(isomin, isomax, num=int(isosurface_count))
    isodata = grid.contour(isosurfaces)

    if smoothing_iterations > 0:
        isodata = isodata.smooth(n_iter=int(smoothing_iterations))

    mesh_data["mesh"] = isodata
    return mesh_data


def bounds_3d(viewer_state):
    return [(viewer_state.z_min, viewer_state.z_max, viewer_state.resolution),
            (viewer_state.y_min, viewer_state.y_max, viewer_state.resolution),
            (viewer_state.x_min, viewer_state.x_max, viewer_state.resolution)]


# For the 3D volume viewer
# This is largely lifted from Luca's plugin
def create_meshes(viewer_state, layer_states, parameters):

    meshes = {}

    bounds = bounds_3d(viewer_state)

    if layer_states is None:
        layer_states = list(viewer_state.layers)

    for layer_state in layer_states:

        if not isinstance(layer_state.layer, GroupedSubset):
            data = layer_state.layer.compute_fixed_resolution_buffer(
                target_data=viewer_state.reference_data,
                bounds=bounds,
                target_cid=layer_state.attribute
            )

            meshes[layer_state.layer.label] = {
                "data": data,
                "color": layer_color(layer_state),
                "opacity": layer_state.alpha,
                "isomin": isomin_for_layer(viewer_state, layer_state),
                "name": layer_state.layer.label
            }

    for layer_state in layer_states:

        if isinstance(layer_state.layer, GroupedSubset):
            parent = layer_state.layer.data
            subcube = parent.compute_fixed_resolution_buffer(
                target_data=viewer_state.reference_data,
                bounds=bounds,
                subset_state=layer_state.layer.subset_state
            )

            datacube = meshes[parent.label]["data"]
            data = subcube * datacube

            meshes[layer_state.layer.label] = {
                "data": data,
                "isomin": isomin_for_layer(viewer_state, layer_state),
                "opacity": layer_state.alpha,
                "color": layer_color(layer_state),
                "name": layer_state.layer.label
            }

            # Delete sublayer data from parent data
            if parent.label in meshes:
                parent_data = meshes[parent.label]["data"]
                parent_data = invert(subcube) * parent_data
                meshes[parent.label]["data"] = parent_data

    for label, item in meshes.items():
        data = item["data"]
        isomin = item["isomin"]

        if parameters[label]["gaussian_filter"]:
            data = gaussian_filter(data, 1)

        # Conventions between pyvista and glue data storage
        data = data.transpose(2, 1, 0)

        grid = pv.ImageData()
        grid.dimensions = (viewer_state.resolution,) * 3
        grid.origin = (viewer_state.x_min, viewer_state.y_min, viewer_state.z_min)
        # Comment from Luca: # I think the voxel spacing will always be 1,
        # because of how glue downsamples to a fixed resolution grid.
        # But don't hold me to this!
        grid.spacing = (1, 1, 1)
        grid.point_data["values"] = data.flatten(order="F")
        isodata = grid.contour([isomin])

        iterations = parameters[label]["smoothing_iterations"]
        if iterations > 0:
            isodata = isodata.smooth(n_iter=iterations)

        item["mesh"] = isodata

    return meshes
