import os

import pyvista as pv

from glue.config import viewer_tool
from glue.viewers.common.tool import Tool

from glue_ar.common import create_plotter
from glue_ar.scatter import scatter_layer_as_multiblock
from glue_ar.export import export_gl_with_alpha, export_modelviewer

__all__ = ["GLTFExportTool"]

# This is just some placeholder image that I found online
AR_ICON = os.path.abspath(os.path.join(os.path.dirname(__file__), "ar"))


# No UI for now - that's easy to add once implementations are complete
# For now, this just exports to "test.gltf" and "test.html"
@viewer_tool
class GLTFExportTool(Tool):
    icon = AR_ICON
    tool_id = "ar:gltf"
    action_text = "Export to glTF"
    tool_tip = "Export the current view to a glTF fileb"

    def activate(self):
        plotter = create_plotter(pv.Plotter.add_mesh, scatter_layer_as_multiblock, self.viewer.state)
        output_filename = "test.glb"
        export_gl_with_alpha(plotter, output_filename)
        export_modelviewer("test.html", output_filename, "Testing visualization")

