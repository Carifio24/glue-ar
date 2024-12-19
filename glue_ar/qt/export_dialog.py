from math import floor, log
import os
from typing import List

from echo import HasCallbackProperties, add_callback
from echo.core import remove_callback
from echo.qt import autoconnect_callbacks_to_qt, connect_checkable_button, connect_value
from glue.core.state_objects import State
from glue_qt.utils import load_ui
from glue_ar.common.export_dialog_base import ARExportDialogBase

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLayout, QSizePolicy, QSlider, QWidget


__all__ = ['QtARExportDialog']


class QtARExportDialog(ARExportDialogBase, QDialog):

    def __init__(self, parent=None, viewer=None):

        ARExportDialogBase.__init__(self, viewer=viewer)
        QDialog.__init__(self, parent=parent)

        self.ui = load_ui('export_dialog.ui', self, directory=os.path.dirname(__file__))

        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)
        self._layer_connections = []
        self._on_layer_change(self.state.layer)

        self.ui.button_cancel.clicked.connect(self.reject)
        self.ui.button_ok.clicked.connect(self.accept)

    def _widgets_for_property(self,
                              instance: HasCallbackProperties,
                              property: str,
                              display_name: str) -> List[QWidget]:
        value = getattr(instance, property)
        t = type(value)
        if t is bool:
            widget = QCheckBox()
            widget.setChecked(value)
            widget.setText(display_name)
            self._layer_connections.append(connect_checkable_button(instance, property, widget))
            return [widget]
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

            def remove_label_callback(widget, update_label=update_label):
                try:
                    remove_callback(instance, property, update_label)
                except ValueError:
                    pass

            update_label(value)
            add_callback(instance, property, update_label)
            widget.destroyed.connect(lambda *args, cb=remove_label_callback: cb(widget))

            steps = round((max - min) / step)
            widget.setMinimum(0)
            widget.setMaximum(steps)
            self._layer_connections.append(connect_value(instance, property, widget, value_range=(min, max)))
            return [label, widget, value_label]
        else:
            return []

    def _clear_layout(self, layout: QLayout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self._clear_layout(item.layout())

                layout.removeItem(item)

            if isinstance(layout, QFormLayout):
                self._clear_form_rows(layout)

    def _clear_form_rows(self, layout: QFormLayout):
        if layout is not None:
            while layout.rowCount():
                layout.removeRow(0)

    def _clear_layer_layout(self):
        self._clear_layout(self.ui.layer_layout)
        self._layer_connections = []

    def _on_layer_change(self, layer_name: str):
        super()._on_layer_change(layer_name)
        multiple_methods = len(self.state.method_helper.choices) > 1
        self.ui.label_method.setVisible(multiple_methods)
        self.ui.combosel_method.setVisible(multiple_methods)

    def _update_layer_ui(self, state: State):
        self._clear_layer_layout()
        for property in state.callback_properties():
            row = QHBoxLayout()
            name = self.display_name(property)
            widgets = self._widgets_for_property(state, property, name)
            for widget in widgets:
                row.addWidget(widget)
            self.ui.layer_layout.addRow(row)

    def _on_filetype_change(self, filetype: str):
        super()._on_filetype_change(filetype)
        gl = filetype.lower() in ("gltf", "glb")
        self.ui.combosel_compression.setVisible(gl)
        self.ui.label_compression_message.setVisible(gl)
        self.ui.bool_modelviewer.setVisible(gl)
        self.ui.label_optimize_for.setVisible(gl)
        self.ui.optimize_for_row.setVisible(gl)
        # self.ui.label_filesize.setVisible(gl)
        # self.ui.label_performance.setVisible(gl)
        # self.ui.slider_optimize_for.setVisible(gl)

    def _on_method_change(self, method_name: str):
        super()._on_method_change(method_name)
        state = self._layer_export_states[self.state.layer][method_name]
        self._update_layer_ui(state)
