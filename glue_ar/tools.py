import os
from os.path import join, split, splitext
from tempfile import NamedTemporaryFile
from threading import Thread

from glue_vispy_viewers.scatter.scatter_viewer import Vispy3DScatterViewerState
from glue_vispy_viewers.volume.layer_state import VolumeLayerState

import ngrok
import pyvista as pv

from qtpy import compat
from qtpy.QtWidgets import QDialog

from glue.config import viewer_tool
from glue.viewers.common.tool import SimpleToolMenu, Tool
from glue_ar.export_dialog import ARExportDialog
from glue_qt.utils.threading import Worker

from glue_ar.qr import get_local_ip
from glue_ar.qr_dialog import QRDialog

from glue_ar.scatter import scatter_layer_as_multiblock
from glue_ar.export import compress_gl, export_gl, export_modelviewer
from glue_ar.exporting_dialog import ExportingDialog
from glue_ar.server import run_ar_server
from glue_ar.utils import bounds_3d_from_layers, xyz_bounds
from glue_ar.volume import bounds_3d, meshes_for_volume_layer, volume_layer_as_multiblock


__all__ = ["ARToolMenu", "ARExportTool", "ARLocalQRTool"]

# This is just some placeholder image that I found online
AR_ICON = os.path.abspath(os.path.join(os.path.dirname(__file__), "ar"))

_FILETYPE_NAMES = {
    ".obj": "OBJ",
    ".gltf": "glTF",
    ".glb": "glB"
}


def create_plotter(viewer, state_dictionary):
    plotter = pv.Plotter()
    layer_states = [layer.state for layer in viewer.layers if layer.enabled and layer.state.visible]
    scatter_viewer = isinstance(viewer.state, Vispy3DScatterViewerState)
    if scatter_viewer:
        bounds = xyz_bounds(viewer.state)
    elif viewer.state.clip_data:
        bounds = bounds_3d(viewer.state)
    else:
        bounds = bounds_3d_from_layers(viewer.state, layer_states)
    frbs = {}
    for layer_state in layer_states:
        layer_info = state_dictionary.get(layer_state.layer.label, {})
        if layer_info:
            layer_info = layer_info.as_dict()
        if isinstance(layer_state, VolumeLayerState):
            meshes = volume_layer_as_multiblock(viewer.state, layer_state,
                                             bounds=bounds,
                                             precomputed_frbs=frbs,
                                             **layer_info)
        else:
            meshes = scatter_layer_as_multiblock(viewer.state, layer_state,
                                                 scaled=scatter_viewer,
                                                 clip_to_bounds=viewer.state.clip_data,
                                                 **layer_info)
        for mesh_info in meshes:
            mesh = mesh_info.pop("mesh")
            if len(mesh.points) > 0:
                plotter.add_mesh(mesh, **mesh_info)

    return plotter


@viewer_tool
class ARToolMenu(SimpleToolMenu):
    tool_id = "ar"
    icon = AR_ICON
    tool_tip = "AR utilities"


@viewer_tool
class ARExportTool(Tool):
    icon = AR_ICON
    tool_id = "save:ar"
    action_text = "Export 3D file"
    tool_tip = "Export the current view to a 3D file"

    _default_filename = "glue_export"

    def activate(self):

        dialog = ARExportDialog(parent=self.viewer, viewer=self.viewer)
        result = dialog.exec_()
        if result == QDialog.Rejected:
            return

        export_path, _ = compat.getsavefilename(parent=self.viewer,
                                                basedir=f"{self._default_filename}.{dialog.state.filetype.lower()}")
        if not export_path:
            return

        _, ext = splitext(export_path)
        filetype = _FILETYPE_NAMES.get(ext, None)

        self._export_to_ar(export_path, dialog.state_dictionary, compress=dialog.state.draco)

        # worker = Worker(self._export_to_ar, export_path, dialog.state_dictionary, compress=dialog.state.draco)
        # exporting_dialog = ExportingDialog(parent=self.viewer, filetype=filetype)
        # worker.result.connect(exporting_dialog.close)
        # worker.error.connect(exporting_dialog.close)
        # worker.start()
        # exporting_dialog.exec_()

    def _export_to_ar(self, filepath, state_dict, compress=True):
        dir, base = split(filepath)
        name, ext = splitext(base)
        plotter = create_plotter(self.viewer, state_dict)
        html_path = join(dir, f"{name}.html")
        if ext in [".gltf", ".glb"]:
            export_gl(plotter, filepath, with_alpha=True, compress=compress)
            if compress:
                compress_gl(filepath)
            export_modelviewer(html_path, base, self.viewer.state.title)
        else:
            plotter.export_obj(filepath)


@viewer_tool
class ARLocalQRTool(Tool):
    icon = AR_ICON
    tool_id = "ar:qr"
    action_text = "3D view via QR"
    tool_tip = "Get a QR code for the current view in 3D"

    def activate(self):
        with NamedTemporaryFile(suffix=".gltf") as gltf_tmp, \
             NamedTemporaryFile(suffix=".html") as html_tmp:

            plotter = create_plotter(self.viewer, {})
            export_gl(plotter, gltf_tmp.name, with_alpha=True)
            _, gltf_base = split(gltf_tmp.name)
            export_modelviewer(html_tmp.name, gltf_base, self.viewer.state.title)

            port = 4000
            directory, filename = split(html_tmp.name)
            server = run_ar_server(port, directory)
            use_ngrok = os.getenv("NGROK_AUTHTOKEN", None) is not None

            try:
                thread = Thread(target=server.serve_forever)
                thread.start()

                if use_ngrok:
                    listener = ngrok.forward(port, authtoken_from_env=True)
                    url = f"{listener.url()}/{filename}"
                else:
                    ip = get_local_ip()
                    url = f"http://{ip}:{port}/{filename}"
                dialog = QRDialog(parent=self.viewer, url=url)
                dialog.exec_()

            finally:
                if use_ngrok:
                    try:
                        ngrok.disconnect(listener.url())
                    except RuntimeError:
                        pass
                server.shutdown()
                server.server_close()
