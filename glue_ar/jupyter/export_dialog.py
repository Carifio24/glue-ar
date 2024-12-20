import ipyvuetify as v  # noqa
from ipyvuetify.VuetifyTemplate import VuetifyTemplate
from ipywidgets import DOMWidget, widget_serialization
import traitlets
from typing import Callable, List, Optional

from echo import HasCallbackProperties
from glue.core.state_objects import State
from glue.viewers.common.viewer import Viewer
from glue_jupyter.link import link
from glue_jupyter.vuetify_helpers import link_glue_choices

from glue_ar.common.export_dialog_base import ARExportDialogBase


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

    layer_layout = traitlets.Instance(v.Container).tag(sync=True, **widget_serialization)
    has_layer_options = traitlets.Bool().tag(sync=True)

    modelviewer = traitlets.Bool(True).tag(sync=True)
    show_modelviewer = traitlets.Bool(True).tag(sync=True)

    def __init__(self,
                 viewer: Viewer,
                 display: Optional[bool] = False,
                 on_cancel: Optional[Callable] = None,
                 on_export: Optional[Callable] = None):
        ARExportDialogBase.__init__(self, viewer=viewer)
        self.layer_layout = v.Container()
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

        widgets = []
        for property, _ in state.iter_callback_properties():
            name = self.display_name(property)
            widgets.extend(self.widgets_for_property(state, property, name))
        self.input_widgets = [w for w in widgets if isinstance(w, v.Slider)]
        self.layer_layout = v.Container(children=widgets, px_0=True, py_0=True)
        self.has_layer_options = len(self.layer_layout.children) > 0

    def _on_method_change(self, method_name: str):
        super()._on_method_change(method_name)
        state = self._layer_export_states[self.state.layer][method_name]
        self._update_layer_ui(state)

    def _on_filetype_change(self, filetype: str):
        super()._on_filetype_change(filetype)
        gl = filetype.lower() in ("gltf", "glb")
        self.show_compression = gl
        self.show_modelviewer = gl

    def widgets_for_property(self,
                             instance: HasCallbackProperties,
                             property: str,
                             display_name: str) -> List[DOMWidget]:

        value = getattr(instance, property)
        t = type(value)
        if t is bool:
            widget = v.Checkbox(label=display_name)
            link((instance, property), (widget, 'value'))
            return [widget]
        elif t in (int, float):
            instance_type = type(instance)
            cb_property = getattr(instance_type, property)
            min = getattr(cb_property, 'min_value', 1 if t is int else 0.01)
            max = getattr(cb_property, 'max_value', 100 * min)
            step = getattr(cb_property, 'resolution', None)
            if step is None:
                step = 1 if t is int else 0.01
            widget = v.Slider(min=min,
                              max=max,
                              step=step,
                              label=display_name,
                              thumb_label=f"{value:g}")
            link((instance, property),
                 (widget, 'v_model'))

            return [widget]
        else:
            return []

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
