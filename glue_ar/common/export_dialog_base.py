from typing import Dict, List, Tuple

from echo import delay_callback
from glue.core.state_objects import State
from glue.viewers.common.viewer import LayerArtist, Viewer
from glue_vispy_viewers.scatter.layer_artist import VispyLayerArtist

from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.export_state import ARExportDialogState
from glue_ar.utils import export_label_for_layer


class ARExportDialogBase:

    def __init__(self, viewer: Viewer):

        self.viewer = viewer
        layers = [layer for layer in self.viewer.layers if layer.enabled and layer.state.visible]
        self.state = ARExportDialogState(layers)

        self._layer_export_states: Dict[str, Dict[str, State]] = {
            export_label_for_layer(layer): {}
            for layer in layers
        }
        self.state_dictionary: Dict[str, Tuple[str, State]] = {}
        self._initialize_dictionaries(layers)

        self.state.add_callback('layer', self._on_layer_change)
        self.state.add_callback('filetype', self._on_filetype_change)
        self.state.add_callback('method', self._on_method_change)
        self.state.add_callback('compression', self._on_compression_change)
        self.state.add_callback('modelviewer', self._on_modelviewer_change)

    def _initialize_dictionaries(self, layers: List[LayerArtist]):
        for layer in layers:
            method = self.state.method
            label = export_label_for_layer(layer)
            if label in self.state_dictionary:
                _, state = self.state_dictionary[label]
            else:
                states = ar_layer_export.export_state_classes(type(layer.state))
                state_cls = next((t[1] for t in states if t[0] == self.state.method), None)
                if state_cls is None:
                    method_names = ar_layer_export.method_names(type(layer.state), self.state.filetype)
                    method = method_names[0]
                    state_cls = next(t[1] for t in states if t[0] == method)
                state = state_cls()
                self.state_dictionary[label] = (method, state)
            self._layer_export_states[label][method] = state

    def _layer_for_label(self, label: str) -> VispyLayerArtist:
        return next(layer for layer in self.state.layers if export_label_for_layer(layer) == label)

    def _update_layer_ui(self, state: State):
        pass

    def _on_layer_change(self, layer_name: str):
        layer = self._layer_for_label(layer_name)
        layer_state_cls = type(layer.state)
        method_names = ar_layer_export.method_names(layer_state_cls, self.state.filetype)
        if layer_name in self.state_dictionary:
            method, state = self.state_dictionary[layer_name]
        else:
            method = method_names[0]
            state = ar_layer_export.options_class(layer_state_cls, method)()
            self.state_dictionary[layer_name] = (method, state)

        with delay_callback(self.state, 'method'):
            method_change = method != self.state.method
            self.state.method_helper.choices = method_names
            self.state.method = method

        if not method_change:
            self._update_layer_ui(state)

    def _on_filetype_change(self, filetype: str):
        pass

    def _on_compression_change(self, compression: str):
        pass

    def _on_modelviewer_change(self, use_modelviewer: bool):
        pass

    def _on_method_change(self, method_name: str):
        if method_name in self._layer_export_states[self.state.layer]:
            state = self._layer_export_states[self.state.layer][method_name]
        else:
            layer = self._layer_for_label(self.state.layer)
            states = ar_layer_export.export_state_classes(type(layer.state))
            state_cls = next(t[1] for t in states if t[0] == method_name)
            state = state_cls()
            self._layer_export_states[self.state.layer][method_name] = state
        self.state_dictionary[self.state.layer] = (method_name, state)

    @staticmethod
    def display_name(prop):
        if prop == "log_points_per_mesh":
            return "Points per mesh"
        return prop.replace("_", " ").capitalize()
