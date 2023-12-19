import os

from qtpy.QtWidgets import QDialog, QListWidgetItem

from glue.core.state_objects import State
from glue.core.data_combo_helper import ComboHelper
from glue_qt.utils import load_ui

from echo import CallbackProperty, SelectionCallbackProperty 
from echo.qt import autoconnect_callbacks_to_qt

__all__ = ["ExportScatterDialog"]


class BaseExportDialogState(State):

    filetype = SelectionCallbackProperty()
    layer = SelectionCallbackProperty()


class BaseExportDialog(QDialog):

    def __init__(self, parent=None, viewer_state=None):

        super(BaseExportDialog, self).__init__(parent=parent)

        self.viewer_state = viewer_state
        self.state = self._state_cls(self.viewer_state)


class ScatterLayerExportState(BaseExportDialogState):

    theta_resolution = CallbackProperty(8)
    phi_resolution = CallbackProperty(8)


# Note that this class only holds the state that is
# currently displayed in the dialog. In particular,
# this means that `theta_resolution` and `phi_resolution`
# represent the resolutions for `layer`
class ExportScatterDialogState(BaseExportDialogState):

    def __init__(self, viewer_state):

        self.layer_state = ExportScatterDialogState(viewer_state)

        super(ExportScatterDialogState, self).__init__()

        self.filetype_helper = ComboHelper(self, 'filetype')
        self.filetype_helper.choices = ['glTF', 'OBJ']

        self.layers = [state for state in viewer_state.layers if state.visible]
        self.layer_helper = ComboHelper(self, 'layer')
        self.layer_helper.choices = [state.layer.label for state in self.layers]


class ExportScatterDialog(QDialog):

    def __init__(self, parent=None, viewer_state=None):

        super(ExportScatterDialog, self).__init__(parent=parent)

        self.viewer_state = viewer_state
        self.state = ExportScatterDialogState(self.viewer_state)
        self.ui = load_ui('export_scatter.ui', self, directory=os.path.dirname(__file__))
        
        layers = [state for state in self.viewer_state.layers if state.visible]
        self.info_dictionary = {
            layer.layer.label: {
                "theta_resolution": 8,
                "phi_resolution": 8
            } for layer in layers
        }

        for layer in layers:
            item = QListWidgetItem(layer.layer.label)
            self.ui.listsel_layer.addItem(item)

        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)

        self.ui.button_cancel.clicked.connect(self.reject)
        self.ui.button_ok.clicked.connect(self.accept)

        self.state.add_callback('theta_resolution', self._on_theta_resolution_change)
        self.state.add_callback('phi_resolution', self._on_phi_resolution_change)
        self.state.add_callback('layer', self._on_layer_change)

    def _on_theta_resolution_change(self, resolution):
        self.info_dictionary[self.state.layer]["theta_resolution"] = int(resolution)

    def _on_phi_resolution_change(self, resolution):
        self.info_dictionary[self.state.layer]["phi_resolution"] = int(resolution)

    def _on_layer_change(self, layer):
        self.state.theta_resolution = self.info_dictionary[layer]["theta_resolution"]
        self.state.phi_resolution = self.info_dictionary[layer]["phi_resolution"]

