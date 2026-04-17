from glue.core.state_objects import State

from glue_ar.common.ranged_callback import RangedCallbackProperty


__all__ = ["ARVispyScatterExportOptions"]


class ARVispyScatterExportOptions(State):
    resolution = RangedCallbackProperty(
            default=3,
            min_value=3,
            max_value=50,
            resolution=1,
            docstring="Controls the resolution of the sphere meshes used for scatter points. "
                      "Higher means better resolution, but a larger filesize.",
    )
    log_points_per_mesh = RangedCallbackProperty(
            default=7,
            min_value=0,
            max_value=7,
            docstring="Controls how many points are put into each mesh. "
                      "Higher means a larger filesize, but better performance."
    )


class ARIpyvolumeScatterExportOptions(State):
    log_points_per_mesh = RangedCallbackProperty(
            default=7,
            min_value=0,
            max_value=7,
            docstring="Controls how many points are put into each mesh. "
                      "Higher means a larger filesize, but better performance."
    )


class ARPointExportOptions(State):
    pass
