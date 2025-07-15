from echo.core import CallbackProperty
from typing import Optional

from glue_ar.utils import clamp, clamp_with_resolution


__all__ = ["RangedCallbackProperty"]


def setup_ranged_callback(state, prop, min_value, max_value, resolution=None):
    def set_adjusted_value(value):
        if resolution is not None:
            adjusted = clamp_with_resolution(value, min_value, max_value, resolution)
        else:
            adjusted = clamp(value, min_value, max_value)
        return adjusted

    state.add_callback(prop, set_adjusted_value, validator=True)


class RangedCallbackProperty(CallbackProperty):

    def __init__(self,
                 default: Optional[float] = None,
                 min_value: float = 0,
                 max_value: float = 1,
                 resolution: Optional[float] = None,
                 **kwargs):
        super().__init__(default=default, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.resolution = resolution

    def __set__(self, instance, value):
        if value is not None:
            if self.resolution is not None:
                value = clamp_with_resolution(value, self.min_value, self.max_value, self.resolution)
            else:
                value = clamp(value, self.min_value, self.max_value)

        super().__set__(instance, value)
