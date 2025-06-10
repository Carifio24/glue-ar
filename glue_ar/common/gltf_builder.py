from __future__ import annotations
from collections import defaultdict

from gltflib import Accessor, AccessorType, AlphaMode, Animation, AnimationSampler, Asset, Attributes, Buffer, \
                    BufferTarget, BufferView, Channel, ComponentType, GLTFModel, \
                    Material, Mesh, Node, PBRMetallicRoughness, Primitive, PrimitiveMode, Scene, \
                    Target
from gltflib.gltf import GLTF
from gltflib.gltf_resource import FileResource
from typing import Dict, Iterable, List, Literal, Optional, Union

from glue_ar.registries import builder


@builder(("gltf", "glb"))
class GLTFBuilder:

    def __init__(self):
        self.materials: List[Material] = []
        self.meshes: List[Mesh] = []
        self.meshes_by_layer: Dict[str, List[int]] = defaultdict(list)
        self.buffers: List[Buffer] = []
        self.buffer_views: List[BufferView] = []
        self.accessors: List[Accessor] = []
        self.file_resources: List[FileResource] = []
        self.channels: List[Channel] = []
        self.samplers: List[AnimationSampler] = []
        self.animations: List[Animation] = []

    def add_material(self,
                     color: Iterable[float],
                     opacity: float = 1,
                     roughness_factor: float = 1,
                     metallic_factor: float = 0,
                     alpha_mode: AlphaMode = AlphaMode.BLEND) -> GLTFBuilder:
        if any(c > 1 for c in color):
            color = [c / 256 for c in color[:3]]
        self.materials.append(
            Material(
                pbrMetallicRoughness=PBRMetallicRoughness(
                    baseColorFactor=list(color[:3]) + [opacity],
                    roughnessFactor=roughness_factor,
                    metallicFactor=metallic_factor
                ),
                alphaMode=alpha_mode.value
            )
        )
        return self

    def add_mesh(self,
                 layer_id: Union[str, Iterable[str]],
                 position_accessor: int,
                 indices_accessor: Optional[int] = None,
                 material: Optional[int] = None,
                 mode: PrimitiveMode = PrimitiveMode.TRIANGLES) -> GLTFBuilder:

        primitive_kwargs = {
                "attributes": Attributes(POSITION=position_accessor),
                "mode": mode
        }
        if indices_accessor is not None:
            primitive_kwargs["indices"] = indices_accessor
        if material is not None:
            primitive_kwargs["material"] = material
        mesh_index = self.mesh_count
        self.meshes.append(
            Mesh(primitives=[
                Primitive(**primitive_kwargs)]
            )
        )
        if isinstance(layer_id, str):
            self.meshes_by_layer[layer_id].append(mesh_index)
        else:
            for id in layer_id:
                self.meshes_by_layer[id].append(mesh_index)
        return self

    def add_buffer(self,
                   byte_length: int,
                   uri: str) -> GLTFBuilder:
        self.buffers.append(
            Buffer(
                byteLength=byte_length,
                uri=uri
            )
        )
        return self

    def add_buffer_view(self,
                        buffer: int,
                        byte_length: int,
                        byte_offset: int,
                        target: Optional[BufferTarget] = None) -> GLTFBuilder:
        self.buffer_views.append(
            BufferView(
                buffer=buffer,
                byteLength=byte_length,
                byteOffset=byte_offset,
                target=target.value if target else None,
            )
        )
        return self

    def add_accessor(self,
                     buffer_view: int,
                     component_type: ComponentType,
                     count: int,
                     type: AccessorType,
                     mins: List[Union[int, float]],
                     maxes: List[Union[int, float]]) -> GLTFBuilder:
        self.accessors.append(
            Accessor(
                bufferView=buffer_view,
                componentType=component_type,
                count=count,
                type=type.value,
                min=mins,
                max=maxes,
            )
        )
        return self

    def add_file_resource(self,
                          filename: str,
                          data: bytearray) -> GLTFBuilder:
        self.file_resources.append(
            FileResource(
                filename,
                data=data,
            )
        )
        return self

    def add_animation(self,
                      name: str,
                      node: int,
                      time_accessor: int,
                      diffs_accessor: int,
                      path: Literal["translation", "rotation", "scale"],
                      interpolation: Literal["STEP", "LINEAR", "CUBICSPLINE"] = "LINEAR") -> GLTFBuilder:

        target = Target(node=node, path=path)
        sampler = AnimationSampler(input=time_accessor, interpolation=interpolation, output=diffs_accessor)
        self.samplers.append(sampler)
        channel = Channel(target=target, sampler=self.sampler_count-1)
        self.channels.append(channel)
        animation = Animation(name=name, channels=[channel], samplers=[sampler])
        self.animations.append(animation)
        return self

    @property
    def material_count(self) -> int:
        return len(self.materials)

    @property
    def mesh_count(self) -> int:
        return len(self.meshes)

    @property
    def buffer_count(self) -> int:
        return len(self.buffers)

    @property
    def buffer_view_count(self) -> int:
        return len(self.buffer_views)

    @property
    def accessor_count(self) -> int:
        return len(self.accessors)

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def sampler_count(self) -> int:
        return len(self.samplers)

    @property
    def animation_count(self) -> int:
        return len(self.animations)

    def build_model(self) -> GLTFModel:
        nodes = [Node(mesh=i) for i in range(len(self.meshes))]
        node_indices = list(range(len(nodes)))
        scenes = [Scene(nodes=node_indices)]
        return GLTFModel(
            asset=Asset(version="2.0"),
            scenes=scenes,
            nodes=nodes,
            meshes=self.meshes,
            buffers=self.buffers,
            bufferViews=self.buffer_views,
            accessors=self.accessors,
            materials=self.materials or None,
            samplers=self.samplers or None,
            animations=self.animations or None,
        )

    def build(self) -> GLTF:
        model = self.build_model()
        return GLTF(model=model, resources=self.file_resources)

    def build_and_export(self, filepath: str):
        self.build().export(filepath)
