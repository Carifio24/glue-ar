from glue.viewers.common.viewer import LayerArtist
from glue_jupyter.ipyvolume.scatter.viewer import Viewer3DStateWidget
from glue_jupyter.view import IPyWidgetView
from glue_jupyter.common.state3d import Scatter3DViewerState
from glue_jupyter.ipyvolume.scatter import Scatter3DLayerState, Scatter3DLayerStateWidget
import ipyreact

# TODO: We'll fill out these more specialized specifics of these classes in the future

class ARViewerState(Scatter3DViewerState):
    pass


class ARScatterLayerState(Scatter3DLayerState):
    pass


class ARScatterLayerStateWidget(Scatter3DLayerStateWidget):
    pass


class ARViewerStateWidget(Viewer3DStateWidget):
    pass


class ARScatterLayerArtist(LayerArtist):
    _layer_state_cls = ARScatterLayerState


# Do the necessary ipyreact setup
ipyreact


class ARWidget(ipyreact.Widget):
    pass


class ARViewer(IPyWidgetView):

    allow_duplicate_data = False
    allow_duplicate_subset = False

    _state_cls = ARViewerState
    _options_cls = ARViewerStateWidget
    _data_artist_cls = ARScatterLayerArtist
    _subset_artist_cls = ARScatterLayerArtist
    _layer_style_widget_class = ARScatterLayerStateWidget

