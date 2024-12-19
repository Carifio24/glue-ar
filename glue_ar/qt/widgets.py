from math import floor, log
from typing import List, Optional, Tuple

from echo import HasCallbackProperties, add_callback, remove_callback
from echo.qt import BaseConnection, connect_checkable_button, connect_value

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLayout, QSizePolicy, QSlider, QWidget


def widgets_for_property(
    instance: HasCallbackProperties,
    property: str,
    display_name: str) -> Tuple[List[QWidget], Optional[BaseConnection]]:
    
    value = getattr(instance, property)
    t = type(value)
    if t is bool:
        widget = QCheckBox()
        widget.setChecked(value)
        widget.setText(display_name)
        return [widget], connect_checkable_button(instance, property, widget)
    elif t in (int, float):
        label = QLabel()
        prompt = f"{display_name}:"
        label.setText(prompt)
        widget = QSlider()
        policy = QSizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        widget.setOrientation(Qt.Orientation.Horizontal)

        widget.setSizePolicy(policy)

        value_label = QLabel()
        instance_type = type(instance)
        cb_property = getattr(instance_type, property)
        min = getattr(cb_property, 'min_value', 1 if t is int else 0.01)
        max = getattr(cb_property, 'max_value', 100 * min)
        step = getattr(cb_property, 'resolution', None)
        if step is None:
            step = 1 if t is int else 0.01
        places = -floor(log(step, 10))

        def update_label(value):
            value_label.setText(f"{value:.{places}f}")

        def remove_label_callback(*args):
            remove_callback(instance, property, update_label)

        update_label(value)
        add_callback(instance, property, update_label)
        widget.destroyed.connect(remove_label_callback)

        steps = round((max - min) / step)
        widget.setMinimum(0)
        widget.setMaximum(steps)

        return [label, widget, value_label], \
                connect_value(instance, property, widget, value_range=(min, max))

    else:
        return [], None

