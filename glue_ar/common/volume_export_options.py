from glue.core.state_objects import State

from glue_ar.common.ranged_callback import RangedCallbackProperty


__all__ = ["ARIsosurfaceExportOptions", "ARVoxelExportOptions"]


class ARIsosurfaceExportOptions(State):
    isosurface_count = RangedCallbackProperty(
        default=20,
        min_value=1,
        max_value=50,
        resolution=1,
        docstring="The number of isosurfaces used in the export.",
    )


class ARVoxelExportOptions(State):
    opacity_cutoff = RangedCallbackProperty(
        default=0.1,
        min_value=0.01,
        max_value=1,
        resolution=0.01,
        docstring="The minimum opacity voxels to retain. Voxels with a lower opacity will be "
                  "omitted from the export.",
    )
    cmap_resolution = RangedCallbackProperty(
        default=0.02,
        min_value=0.005,
        max_value=1,
        resolution=0.005,
        docstring="The resolution of the color and opacity in the exported figure. Values will be "
                  "rounded to the nearest integer multiple of this value.",
    )
    opacity_factor = RangedCallbackProperty(
        default=1,
        min_value=0,
        max_value=2,
        resolution=0.01,
        docstring="An overall factor by which to adjust the opacities of output voxels."
    )

    # log_voxels_per_mesh = RangedCallbackProperty(
    #     default=7,
    #     min_value=0,
    #     max_value=7,
    #     docstring="Controls how many voxels are put into each mesh. "
    #               "Higher means a larger filesize, but better performance."
    # )


class ARVolumeExportOptions(State):
    pass
