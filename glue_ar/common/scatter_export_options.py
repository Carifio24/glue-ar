from glue.core.state_objects import State

from glue_ar.common.ranged_callback import RangedCallbackProperty


__all__ = ["ARVispyScatterExportOptions"]


class ARVispyScatterExportOptions(State):
    resolution = RangedCallbackProperty(default=10, min_value=3, max_value=50, resolution=1)
    optimize_for_performance = RangedCallbackProperty(default=6, min_value=0, max_value=6, resolution=1)


class ARIpyvolumeScatterExportOptions(State):
    optimize_for_performance = RangedCallbackProperty(default=6, min_value=0, max_value=6, resolution=1)
