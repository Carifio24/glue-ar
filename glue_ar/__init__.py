from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


from .common.gltf_builder import GLTFBuilder  # noqa: F401
from .common.usd_builder import USDBuilder  # noqa: F401
from .common.stl_builder import STLBuilder  # noqa: F401
from .compression import compress_draco, compress_meshoptimizer  # noqa: F401


def setup_qt():
    try:
        from glue_vispy_viewers.scatter.qt.scatter_viewer import VispyScatterViewer
    except ImportError:
        from glue_vispy_viewers.scatter.scatter_viewer import VispyScatterViewer

    from .qt.export_tool import QtARExportTool  # noqa

    VispyScatterViewer.subtools = {
        **VispyScatterViewer.subtools,
        "save": VispyScatterViewer.subtools["save"] + ["save:ar"]
    }

    try:
        from glue_vispy_viewers.volume.qt.volume_viewer import VispyVolumeViewer
    except ImportError:
        from glue_vispy_viewers.volume.volume_viewer import VispyVolumeViewer

    VispyVolumeViewer.subtools = {
        **VispyVolumeViewer.subtools,
        "save": VispyVolumeViewer.subtools["save"] + ["save:ar"]
    }

    try:
        from .qt.qr_tool import ARLocalQRTool  # noqa
        VispyScatterViewer.tools = [t for t in VispyScatterViewer.tools] + ["ar"]
        VispyVolumeViewer.tools = [t for t in VispyVolumeViewer.tools] + ["ar"]
        VispyScatterViewer.subtools["ar"] = ["ar:qr"]
        VispyVolumeViewer.subtools["ar"] = ["ar:qr"]
    except ImportError:
        pass


def setup_jupyter():
    from .jupyter.export_tool import JupyterARExportTool  # noqa
    try:
        from glue_vispy_viewers.scatter.jupyter import JupyterVispyScatterViewer
        from glue_vispy_viewers.volume.jupyter import JupyterVispyVolumeViewer
        JupyterVispyScatterViewer.tools = [t for t in JupyterVispyScatterViewer.tools] + ["save:ar_jupyter"]
        JupyterVispyVolumeViewer.tools = [t for t in JupyterVispyVolumeViewer.tools] + ["save:ar_jupyter"]
    except ImportError:
        pass

    from glue_jupyter.ipyvolume.scatter import IpyvolumeScatterView
    from glue_jupyter.ipyvolume.volume import IpyvolumeVolumeView
    IpyvolumeScatterView.tools = [t for t in IpyvolumeScatterView.tools] + ["save:ar_jupyter"]
    IpyvolumeVolumeView.tools = [t for t in IpyvolumeVolumeView.tools] + ["save:ar_jupyter"]


def setup():

    try:
        setup_qt()
    except ImportError:
        pass

    try:
        setup_jupyter()
    except ImportError:
        pass
