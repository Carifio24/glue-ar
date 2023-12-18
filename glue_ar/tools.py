import os
from os.path import basename, splitext

import pyvista as pv

from qtpy import compat
from qtpy.QtWidgets import QDialog

from glue.config import viewer_tool
from glue.viewers.common.tool import Tool

from glue_ar.export_scatter import ExportScatterDialog
from glue_ar.scatter import scatter_layer_as_multiblock 
from glue_ar.export import export_gl, export_modelviewer
from glue_ar.volume import create_meshes

try:
    from glue_wwt.viewer.table_layer import WWTTableLayerState 
except ImportError:
    WWTTableLayerState = type(None)

__all__ = ["GLScatterExportTool", "GLVolumeExportTool"]

# This is just some placeholder image that I found online
AR_ICON = os.path.abspath(os.path.join(os.path.dirname(__file__), "ar"))


# No UI for now - that's easy to add once implementations are complete
# For now, this just exports to "test.gltf" and "test.html"
@viewer_tool
class GLScatterExportTool(Tool):
    icon = AR_ICON
    tool_id = "ar:scatter-gl"
    action_text = "Export to gl"
    tool_tip = "Export the current view to a glB file"

    def activate(self):
 
        dialog = ExportScatterDialog(parent=self.viewer, viewer_state=self.viewer.state)
        result = dialog.exec_()
        if result == QDialog.Rejected:
            return

        export_path, _ = compat.getsavefilename(parent=self.viewer, basedir=f"scatter.{dialog.state.filetype}".lower())
        if not export_path:
            return

        plotter = pv.Plotter()
        layer_states = [state for state in self.viewer.state.layers if state.visible]
        for layer_state in layer_states:
            layer_info = dialog.info_dictionary[layer_state.layer.label]
            mesh_info = scatter_layer_as_multiblock(self.viewer.state, layer_state, **layer_info)
            data = mesh_info.pop("data")
            plotter.add_mesh(data, **mesh_info)

        base = basename(export_path)
        name, _ = splitext(base)
        html_path = f"{name}.html"
        export_gl(plotter, export_path, with_alpha=True)
        
        export_modelviewer(html_path, base, "Testing visualization")


@viewer_tool
class GLVolumeExportTool(Tool):
    icon = AR_ICON
    tool_id = "ar:volume-gl"
    action_text = "Export to gl"
    tool_tip = "Export the current view to a glB file"

    def activate(self):
        plotter = pv.Plotter()
        meshes = create_meshes(self.viewer.state, use_gaussian_filter=True, smoothing_iteration_count=10)
        for data in meshes.values():
            mesh = data.pop("mesh")
            plotter.add_mesh(mesh, color=data["color"], opacity=data["opacity"])
        export_gl(plotter, "volume.gltf", with_alpha=True)  # Do we want alpha for volume renderings?
        export_modelviewer("volume.html", "volume.gltf", "Testing visualization")


@viewer_tool
class GLWWT3DExportTool(Tool):
    icon = AR_ICON
    tool_id = "ar:wwt-3d-gl"
    action_text = "Export to gl"
    tool_tip = "Export the table layers of the current view to a gl file"

    def activate(self):

        from glue_ar.scatter import wwt_table_layer_as_multiblock

        dialog = ExportScatterDialog(parent=self.viewer, viewer_state=self.viewer.state)
        result = dialog.exec_()
        if result == QDialog.Rejected:
            return

        export_path, _ = compat.getsavefilename(parent=self.viewer, basedir=f"scatter.{dialog.state.filetype}".lower())
        if not export_path:
            return

        plotter = pv.Plotter()
        layer_states = [state for state in self.viewer.state.layers if state.visible and isinstance(state, WWTTableLayerState)]
        for layer_state in layer_states:
            layer_info = dialog.info_dictionary[layer_state.layer.label]
            mesh_info = wwt_table_layer_as_multiblock(self.viewer.state, layer_state, **layer_info)
            data = mesh_info.pop("data")
            plotter.add_mesh(data, **mesh_info)

        base = basename(export_path)
        name, _ = splitext(base)
        html_path = f"{name}.html"
        export_gl(plotter, export_path, with_alpha=True)
        
        export_modelviewer(html_path, base, "Testing visualization")


