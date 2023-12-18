from glue_ar.tools import *

def setup():
    from glue_qt.config import qt_client
    
    from glue_vispy_viewers.scatter.scatter_viewer import VispyScatterViewer
    VispyScatterViewer.tools += ["ar:scatter-gl"]

    from glue_vispy_viewers.volume.volume_viewer import VispyVolumeViewer
    VispyVolumeViewer.tools += ["ar:volume-gl"]

    try:
        from glue_wwt.viewer.qt_data_viewer import WWTQtViewer
        WWTQtViewer.tools += ["ar:wwt-3d-gl"]
    except ImportError:
        pass
