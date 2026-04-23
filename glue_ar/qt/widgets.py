from math import floor, log
from os.path import join
from typing import Tuple

from echo import CallbackProperty, HasCallbackProperties, add_callback, remove_callback
from echo.qt import BaseConnection, connect_checkable_button, connect_value
from qtpy.QtGui import QCursor, QEnterEvent, QIcon
from qtpy.QtCore import Qt, QEvent
from qtpy.QtWidgets import QCheckBox, QPushButton, QSpacerItem, QToolTip, QLabel, QSizePolicy, QSlider, QWidget

from glue_ar.utils import RESOURCES_DIR


def info_tooltip(cb_property: CallbackProperty) -> str:
    return f"<qt>{cb_property.__doc__ or ''}</qt>"


def info_enter_event_handler(_event: QEnterEvent, cb_property: CallbackProperty):
    # Make the tooltip be rich text so that it will line wrap
    QToolTip.showText(QCursor.pos(), info_tooltip(cb_property))


def info_leave_event_handler(_event: QEvent):
    QToolTip.hideText()


def horizontal_spacer(width: int = 40, height: int = 20) -> QSpacerItem:
    return QSpacerItem(width, height, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)


def info_button(cb_property: CallbackProperty) -> QPushButton:
    button = QPushButton()
    button.setCheckable(False)
    button.setFlat(True)
    icon_path = join(RESOURCES_DIR, "info.png")
    icon = QIcon(icon_path)
    button.setIcon(icon)
    button.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))

    # We want the tooltip to show immediately, rather than have a delay
    if cb_property.__doc__:
        button.enterEvent = lambda event: info_enter_event_handler(event, cb_property=cb_property)
        button.leaveEvent = info_leave_event_handler

    return button


def boolean_callback_widgets(instance: HasCallbackProperties,
                             property: str,
                             display_name: str,
                             **kwargs) -> Tuple[Tuple[Tuple[QWidget]], connect_checkable_button]:

    value = getattr(instance, property)
    instance_type = type(instance)
    cb_property: CallbackProperty = getattr(instance_type, property)

    checkbox = QCheckBox()
    checkbox.setChecked(value)
    checkbox.setText(display_name)
    connection = connect_checkable_button(instance, property, checkbox)
    if cb_property.__doc__:
        spacer = horizontal_spacer(width=40, height=20)
        button = info_button(cb_property)
        return ((checkbox, spacer, button),), connection
    else:
        return ((checkbox,),), connection


def number_callback_widgets(instance: HasCallbackProperties,
                            property: str,
                            display_name: str,
                            label_for_value=True,
                            **kwargs) -> Tuple[Tuple[Tuple[QWidget]], connect_value]:

    value = getattr(instance, property)
    instance_type = type(instance)
    cb_property: CallbackProperty = getattr(instance_type, property)
    t = type(value)

    label = QLabel()
    prompt = f"{display_name}:"
    label.setText(prompt)

    slider = QSlider()
    policy = QSizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
    policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
    slider.setOrientation(Qt.Orientation.Horizontal)
    slider.setSizePolicy(policy)

    min = getattr(cb_property, 'min_value', 1 if t is int else 0.01)
    max = getattr(cb_property, 'max_value', 100 * min)
    step = getattr(cb_property, 'resolution', None)
    if step is None:
        step = 1 if t is int else 0.01
    places = -floor(log(step, 10))

    if label_for_value:
        value_label = QLabel()

        def update_label(value):
            value_label.setText(f"{value:.{places}f}")

        def remove_label_callback(widget, update_label=update_label):
            try:
                remove_callback(instance, property, update_label)
            except ValueError:
                pass

        def on_widget_destroyed(widget, cb=remove_label_callback):
            cb(widget)

        update_label(value)
        add_callback(instance, property, update_label)
        slider.destroyed.connect(on_widget_destroyed)

        value_widgets = (slider, value_label)
    else:
        value_widgets = (slider,)

    steps = round((max - min + 1) / step)
    slider.setMinimum(0)
    slider.setMaximum(steps)
    print(max, min, step, steps)
    connection = connect_value(instance, property, slider, value_range=(min, max))

    if cb_property.__doc__:
        button = info_button(cb_property)
        spacer = horizontal_spacer(width=40, height=20)
        return ((label, spacer, button), value_widgets), connection
    else:
        return ((label,), value_widgets), connection


def widgets_for_callback_property(instance: HasCallbackProperties,
                                  property: str,
                                  display_name: str,
                                  **kwargs) -> Tuple[Tuple[Tuple[QWidget]], BaseConnection]:

    t = type(getattr(instance, property))
    if t is bool:
        return boolean_callback_widgets(instance, property, display_name, **kwargs)
    elif t in (int, float):
        return number_callback_widgets(instance, property, display_name, **kwargs)
    else:
        raise ValueError("Unsupported callback property type!")
