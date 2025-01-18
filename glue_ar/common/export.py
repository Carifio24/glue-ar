from collections import defaultdict
from os.path import extsep, join, split, splitext
from string import Template
from typing import Dict, Optional
from glue.core.state_objects import State
from glue.config import settings
from glue.viewers.common.state import LayerState
from glue_vispy_viewers.scatter.viewer_state import Vispy3DViewerState
from glue_vispy_viewers.volume.layer_state import VolumeLayerState


from glue_ar.common.export_options import ar_layer_export
from glue_ar.registries import builder as builder_registry, compressor as compressor_registry
from glue_ar.utils import PACKAGE_DIR, RESOURCES_DIR, Bounds, BoundsWithResolution, export_label_for_layer

from typing import List, Tuple, Union


NODE_MODULES_DIR = join(PACKAGE_DIR, "js", "node_modules")


def export_viewer(viewer_state: Vispy3DViewerState,
                  layer_states: List[LayerState],
                  bounds: Union[Bounds, BoundsWithResolution],
                  state_dictionary: Dict[str, Tuple[str, State]],
                  filepath: str,
                  compression: Optional[str] = "None",
                  model_viewer: bool = False):

    base, ext = splitext(filepath)
    ext = ext[1:]
    builder_cls = builder_registry.members.get(ext)
    builder = builder_cls()
    layer_groups = defaultdict(list)
    export_groups = defaultdict(list)
    for layer_state in layer_states:
        name, export_state = state_dictionary[export_label_for_layer(layer_state)]
        key = (type(layer_state), name)
        layer_groups[key].append(layer_state)
        export_groups[key].append(export_state)

    for key, states in layer_groups.items():
        export_states = export_groups[key]
        layer_state_cls, name = key
        spec = ar_layer_export.export_spec(layer_state_cls, name, ext)
        if spec.multiple:
            spec.export_method(builder, viewer_state, states, export_states, bounds)
        else:
            for layer_state, export_state in zip(states, export_states):
                spec.export_method(builder, viewer_state, layer_state, export_state, bounds)

    builder.build_and_export(filepath)

    if ext in ("gltf", "glb"):
        if (compression is not None) and (compression != "None"):
            compress_gl(filepath, method=compression)
        if model_viewer:
            mv_path = f"{base}{extsep}html"
            export_modelviewer(mv_path, filepath, layer_states, viewer_state.title)


def compress_gl(filepath: str, method: str = "draco"):
    compressor = compressor_registry.members.get(method.lower(), None)
    if compressor is None:
        raise ValueError("Invalid compression method specified")
    compressor(filepath)


def export_modelviewer(output_path: str,
                       gltf_path: str,
                       layer_states: List[LayerState],
                       alt_text: str):
    mv_url = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"
    with open(join(RESOURCES_DIR, "model-viewer.html")) as f:
        html_template = f.read()
    with open(join(RESOURCES_DIR, "model-viewer.css")) as g:
        css_template = g.read()
    css = Template(css_template).substitute({"bg_color": settings.BACKGROUND_COLOR})
    with open(join(RESOURCES_DIR, "model-viewer.js")) as h:
        javascript = h.read()

    buttons = [f"<button data-layer=\"{index}\">{state.layer.label}</button>" for index, state in enumerate(layer_states)]
    buttons = "\n".join(buttons)

    _, gltf_name = split(gltf_path)

    substitutions = {
        "url": mv_url,
        "gltf_path": gltf_name,
        "alt_text": alt_text,
        "style": css,
        "button_text": "View in AR",
        "buttons": buttons,
        "script": javascript,
    }
    html = Template(html_template).substitute(substitutions)

    with open(output_path, 'w') as of:
        of.write(html)
