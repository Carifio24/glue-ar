from collections import defaultdict
from math import floor
from os.path import extsep, join, split, splitext
from string import Template
from typing import Dict, Optional
from glue.core.state_objects import State
from glue.config import settings
from glue.viewers.common.state import LayerState
from glue_vispy_viewers.scatter.viewer_state import Vispy3DViewerState


from glue_ar.common.export_options import ar_layer_export
from glue_ar.common.gltf_builder import GLTFBuilder
from glue_ar.registries import builder as builder_registry, compressor as compressor_registry
from glue_ar.utils import PACKAGE_DIR, RESOURCES_DIR, Bounds, BoundsWithResolution, export_label_for_layer, rgb_to_hex

from typing import List, Tuple, Union


NODE_MODULES_DIR = join(PACKAGE_DIR, "js", "node_modules")


def export_viewer(viewer_state: Vispy3DViewerState,
                  layer_states: List[LayerState],
                  bounds: Union[Bounds, BoundsWithResolution],
                  state_dictionary: Dict[str, Tuple[str, State]],
                  filepath: str,
                  allow_multiple: Optional[bool] = True,
                  compression: Optional[str] = "None",
                  model_viewer: bool = False,
                  layer_controls: bool = True):

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
        if spec.multiple and allow_multiple:
            spec.export_method(builder, viewer_state, states, export_states, bounds)
        else:
            for layer_state, export_state in zip(states, export_states):
                spec.export_method(builder, viewer_state, layer_state, export_state, bounds)

    builder.build_and_export(filepath)

    if ext in ("gltf", "glb"):
        # We can only add layer controls if we aren't using compression
        layer_controls = layer_controls and compression == "None"
        if (compression is not None) and (compression != "None"):
            compress_gl(filepath, method=compression)
        if model_viewer:
            mv_path = f"{base}{extsep}html"
            export_modelviewer(output_path=mv_path,
                               gltf_path=filepath,
                               builder=builder,
                               alt_text=viewer_state.title,
                               layer_controls=layer_controls)


def compress_gl(filepath: str, method: str = "draco"):
    compressor = compressor_registry.members.get(method.lower(), None)
    if compressor is None:
        raise ValueError("Invalid compression method specified")
    compressor(filepath)


def export_modelviewer(output_path: str,
                       gltf_path: str,
                       builder: GLTFBuilder,
                       alt_text: str,
                       layer_controls: bool = True):
    mv_url = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"
    with open(join(RESOURCES_DIR, "model-viewer.html")) as f:
        html_template = f.read()
    with open(join(RESOURCES_DIR, "model-viewer.css")) as g:
        css_template = g.read()
    css = Template(css_template).substitute({"bg_color": settings.BACKGROUND_COLOR})
    with open(join(RESOURCES_DIR, "model-viewer.js")) as h:
        javascript = h.read()

    if layer_controls:
        controls = ["<h3>Toggle Layers</h3>"]
        for index, (layer_id, mesh_indices) in enumerate(builder.meshes_by_layer.items()):
            meshes_string = ",".join(str(idx) for idx in mesh_indices)
            color_mesh_index = mesh_indices[floor(len(mesh_indices) / 2)]
            material_index = builder.meshes[color_mesh_index].primitives[0].material or 0
            color = rgb_to_hex(*builder.materials[material_index].pbrMetallicRoughness.baseColorFactor[:3])
            controls.append(f"<button data-color=\"{color}\" data-layer=\"{index}\" data-meshes=\"{meshes_string}\">{layer_id}</button>")
        controls = "\n".join(controls)
    else:
        controls = ""

    _, gltf_name = split(gltf_path)

    substitutions = {
        "url": mv_url,
        "gltf_path": gltf_name,
        "alt_text": alt_text,
        "style": css,
        "button_text": "View in AR",
        "controls": controls,
        "script": javascript,
    }
    html = Template(html_template).substitute(substitutions)

    with open(output_path, 'w') as of:
        of.write(html)
