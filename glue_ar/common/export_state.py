from echo import SelectionCallbackProperty
from glue.core.data_combo_helper import ComboHelper
from glue.core.state_objects import State
from glue_vispy_viewers.common.layer_state import LayerState

from glue_ar.utils import export_label_for_layer

from typing import Iterable


__all__ = ["ARExportDialogState"]


class ARExportDialogState(State):

    filetype = SelectionCallbackProperty()
    layer = SelectionCallbackProperty()
    compression = SelectionCallbackProperty()
    method = SelectionCallbackProperty()

    def __init__(self, layers: Iterable[LayerState]):

        super(ARExportDialogState, self).__init__()

        self.filetype_helper = ComboHelper(self, 'filetype')
        self.filetype_helper.choices = ['glB', 'glTF', 'USDZ', 'USDC', 'USDA', 'STL']

        self.compression_helper = ComboHelper(self, 'compression')
        self.compression_helper.choices = ['None', 'Draco', 'Meshoptimizer']

        self.method_helper = ComboHelper(self, 'method')

        self.layers = layers
        self.layer_helper = ComboHelper(self, 'layer')
        self.layer_helper.choices = [export_label_for_layer(layer_state) for layer_state in layers]
