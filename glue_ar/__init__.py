from pkg_resources import get_distribution, DistributionNotFound
import traceback

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


def setup_qt():
    try:
        from glue_vispy_viewers.scatter.qt.scatter_viewer import VispyScatterViewer
    except (ModuleNotFoundError, ImportError):
        from glue_vispy_viewers.scatter.scatter_viewer import VispyScatterViewer

    from .qt.export_tool import QtARExportTool  # noqa

    VispyScatterViewer.subtools = {
        **VispyScatterViewer.subtools,
        "save": VispyScatterViewer.subtools["save"] + ["save:ar"]
    }

    try:
        from glue_vispy_viewers.volume.qt.volume_viewer import VispyVolumeViewer
    except (ModuleNotFoundError, ImportError):
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
    except (ModuleNotFoundError, ImportError):
        pass


def setup_jupyter():
    from .jupyter.export_tool import JupyterARExportTool  # noqa
    try:
        from glue_vispy_viewers.scatter.jupyter import JupyterVispyScatterViewer
        from glue_vispy_viewers.volume.jupyter import JupyterVispyVolumeViewer
        JupyterVispyScatterViewer.tools = [t for t in JupyterVispyScatterViewer.tools] + ["save:ar_jupyter"]
        JupyterVispyVolumeViewer.tools = [t for t in JupyterVispyVolumeViewer.tools] + ["save:ar_jupyter"]
    except (ModuleNotFoundError, ImportError):
        pass

    from glue_jupyter.ipyvolume.scatter import IpyvolumeScatterView
    from glue_jupyter.ipyvolume.volume import IpyvolumeVolumeView
    IpyvolumeScatterView.tools = [t for t in IpyvolumeScatterView.tools] + ["save:ar_jupyter"]
    IpyvolumeVolumeView.tools = [t for t in IpyvolumeVolumeView.tools] + ["save:ar_jupyter"]


def setup():
    try:
        setup_qt()
    except (ModuleNotFoundError, ImportError):
        print("Qt setup error")
        print(traceback.format_exc())
        pass

    try:
        setup_jupyter()
    except (ModuleNotFoundError, ImportError):
        print("Jupyter setup error")
        print(traceback.format_exc())
        pass
