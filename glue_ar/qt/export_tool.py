from os.path import splitext

from qtpy import compat
from qtpy.QtWidgets import QDialog

from glue.config import viewer_tool
from glue.viewers.common.tool import SimpleToolMenu, Tool
from glue_qt.utils.threading import Worker

from glue_ar.utils import AR_ICON, is_volume_viewer, xyz_bounds
from glue_ar.common.export import export_viewer
from glue_ar.qt.export_dialog import QtARExportDialog
from glue_ar.qt.exporting_dialog import ExportingDialog


__all__ = ["ARToolMenu", "QtARExportTool"]

_FILETYPE_NAMES = {
    "gltf": "glTF",
    "glb": "glB",
    "usdz": "USDZ",
    "usdc": "USDC",
    "usda": "USDA",
}


@viewer_tool
class ARToolMenu(SimpleToolMenu):
    tool_id = "ar"
    icon = AR_ICON
    tool_tip = "AR utilities"


@viewer_tool
class QtARExportTool(Tool):
    icon = AR_ICON
    tool_id = "save:ar"
    action_text = "Export 3D file"
    tool_tip = "Export the current view to a 3D file"

    _default_filename = "glue_export"

    def activate(self):

        dialog = QtARExportDialog(parent=self.viewer, viewer=self.viewer)
        result = dialog.exec_()
        if result == QDialog.Rejected:
            return

        export_path, _ = compat.getsavefilename(parent=self.viewer,
                                                basedir=f"{self._default_filename}.{dialog.state.filetype.lower()}")
        if not export_path:
            return

        layer_states = [layer.state for layer in self.viewer.layers if
                        layer.enabled and layer.state.visible]
        bounds = xyz_bounds(self.viewer.state, with_resolution=is_volume_viewer(self.viewer))

        # self._start_worker(export_viewer,
        export_viewer(
                           viewer_state=self.viewer.state,
                           layer_states=layer_states,
                           bounds=bounds,
                           state_dictionary=dialog.state_dictionary,
                           filepath=export_path,
                           compression=dialog.state.compression)

    def _start_worker(self, exporter, **kwargs):
        _, ext = splitext(kwargs["filepath"])
        ext = ext[1:]
        filetype = _FILETYPE_NAMES.get(ext, None)
        worker = Worker(exporter, **kwargs)
        exporting_dialog = ExportingDialog(parent=self.viewer, filetype=filetype)
        worker.result.connect(exporting_dialog.close)
        worker.error.connect(exporting_dialog.close)
        worker.start()
        exporting_dialog.exec_()
