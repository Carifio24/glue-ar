import os

from echo.qt import autoconnect_callbacks_to_qt
from glue.core.state_objects import State
from glue_qt.utils import load_ui
from glue_ar.common.export_dialog_base import ARExportDialogBase

from qtpy.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLayoutItem, QVBoxLayout, QLayout, QWidget

from glue_ar.qt.widgets import widgets_for_callback_property


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
        self.ui.line_2.setVisible(multiple_methods)

    def _update_layer_ui(self, state: State):
        self._clear_layer_layout()
        for property in state.callback_properties():
            is_log_pm = (property in ("log_points_per_mesh", "log_voxels_per_mesh"))
            # TODO: Think of a cleaner way to handle this
            if is_log_pm and self.state.filetype.lower() not in ("gltf", "glb"):
                continue
            row = QVBoxLayout()
            name = self.display_name(property)
            widget_tuples, connection = widgets_for_callback_property(state, property, name,
                                                                      label_for_value=not is_log_pm)
            self._layer_connections.append(connection)
            for widgets in widget_tuples:
                subrow = QHBoxLayout()
                for widget in widgets:
                    if isinstance(widget, QWidget):
                        subrow.addWidget(widget)
                    elif isinstance(widget, QLayoutItem):
                        subrow.addItem(widget)
                row.addLayout(subrow)
            self.ui.layer_layout.addLayout(row)

    def _on_filetype_change(self, filetype: str):
        super()._on_filetype_change(filetype)
        state = self._layer_export_states[self.state.layer][self.state.method]
        self._update_layer_ui(state)
        gl = filetype.lower() in ("gltf", "glb")
        self.ui.combosel_compression.setVisible(gl)
        self.ui.label_compression_message.setVisible(gl)
        self.ui.bool_modelviewer.setVisible(gl)
        self.ui.bool_layer_controls.setVisible(gl)

    def _on_method_change(self, method_name: str):
        super()._on_method_change(method_name)
        state = self._layer_export_states[self.state.layer][method_name]
        self._update_layer_ui(state)

    def _on_compression_change(self, compression: str):
        self.ui.layer_controls.setEnabled(
            self.state.modelviewer and
            compression == "None"
        )

    def _on_modelviewer_change(self, use_modelviewer: bool):
        self.ui.bool_layer_controls.setEnabled(
            self.state.compression == "None" and
            use_modelviewer
        )
