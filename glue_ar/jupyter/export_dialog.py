import ipyvuetify as v  # noqa
from ipyvuetify.VuetifyTemplate import VuetifyTemplate
from ipywidgets import widget_serialization
import traitlets
from typing import Callable, Optional

from glue.core.state_objects import State
from glue.viewers.common.viewer import Viewer
from glue_jupyter.link import link
from glue_jupyter.vuetify_helpers import link_glue_choices

from glue_ar.common.export_dialog_base import ARExportDialogBase
from glue_ar.jupyter.widgets import widgets_for_callback_property


class JupyterARExportDialog(ARExportDialogBase, VuetifyTemplate):

    template_file = (__file__, "export_dialog.vue")
    dialog_open = traitlets.Bool().tag(sync=True)

    layer_items = traitlets.List().tag(sync=True)
    layer_selected = traitlets.Int().tag(sync=True)

    compression_items = traitlets.List().tag(sync=True)
    compression_selected = traitlets.Int().tag(sync=True)
    show_compression = traitlets.Bool(True).tag(sync=True)

    filetype_items = traitlets.List().tag(sync=True)
    filetype_selected = traitlets.Int().tag(sync=True)

    method_items = traitlets.List().tag(sync=True)
    method_selected = traitlets.Int().tag(sync=True)

    layer_layout = traitlets.Instance(v.Col).tag(sync=True, **widget_serialization)
    has_layer_options = traitlets.Bool().tag(sync=True)

    modelviewer = traitlets.Bool(True).tag(sync=True)
    show_modelviewer = traitlets.Bool(True).tag(sync=True)

    def __init__(self,
                 viewer: Viewer,
                 display: Optional[bool] = False,
                 on_cancel: Optional[Callable] = None,
                 on_export: Optional[Callable] = None):

        ARExportDialogBase.__init__(self, viewer=viewer)
        self.layer_layout = v.Col()
        VuetifyTemplate.__init__(self)

        self._on_layer_change(self.state.layer)

        link_glue_choices(self, self.state, 'layer')
        link_glue_choices(self, self.state, 'compression')
        link_glue_choices(self, self.state, 'filetype')
        link_glue_choices(self, self.state, 'method')
        link((self, 'modelviewer'), (self.state, 'modelviewer'))

        self.dialog_open = display
        self.on_cancel = on_cancel
        self.on_export = on_export

        self.input_widgets = []

    def _update_layer_ui(self, state: State):
        if self.layer_layout is not None:
            for widget in self.layer_layout.children:
                widget.close()

        rows = []
        input_widgets = []
        self.layer_layout = v.Col()
        for property, _ in state.iter_callback_properties():
            is_log_pm = (property in ("log_points_per_mesh", "log_voxels_per_mesh"))
            # TODO: Think of a cleaner way to handle this
            if is_log_pm and self.state.filetype.lower() not in ("gltf", "glb"):
                continue
            name = self.display_name(property)
            widgets = widgets_for_callback_property(state, property, name, label_for_value=not is_log_pm)
            input_widgets.extend(w for w in widgets if isinstance(w, v.Slider))
            rows.append(v.Row(children=widgets, align="center"))

        self.layer_layout.children = rows
        self.input_widgets = input_widgets
        self.has_layer_options = len(self.layer_layout.children) > 0

    def _on_method_change(self, method_name: str):
        super()._on_method_change(method_name)
        state = self._layer_export_states[self.state.layer][method_name]
        self._update_layer_ui(state)

    def _on_filetype_change(self, filetype: str):
        super()._on_filetype_change(filetype)
        state = self._layer_export_states[self.state.layer][self.state.method]
        self._update_layer_ui(state)
        gl = filetype.lower() in ("gltf", "glb")
        self.show_compression = gl
        self.show_modelviewer = gl

    def _on_compression_change(self, compression: str):
        pass

    def vue_cancel_dialog(self, *args):
        self.state_dictionary = {}
        self.dialog_open = False
        if self.on_cancel:
            self.on_cancel()

    def vue_export_viewer(self, *args):
        okay = all(not widget.error for widget in self.input_widgets)
        if not okay:
            return
        self.dialog_open = False
        if self.on_export:
            self.on_export()
