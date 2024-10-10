from gltflib import Accessor, AccessorType, AlphaMode, Asset, Attributes, Buffer, \
                    BufferTarget, BufferView, ComponentType, GLTFModel, \
                    Material, Mesh, Node, PBRMetallicRoughness, Primitive, PrimitiveMode, Scene
from gltflib.gltf import GLTF
from gltflib.gltf_resource import FileResource
from typing import Iterable, List, Optional, Union


class GLTFBuilder:

    def __init__(self):
        self.materials: List[Material] = []
        self.meshes: List[Mesh] = []
        self.buffers: List[Buffer] = []
        self.buffer_views: List[BufferView] = []
        self.accessors: List[Accessor] = []
        self.file_resources: List[FileResource] = []

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
        self.meshes.append(
            Mesh(primitives=[
                Primitive(**primitive_kwargs)]
            )
        )
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
                        target: BufferTarget) -> GLTFBuilder:
        self.buffer_views.append(
            BufferView(
                buffer=buffer,
                byteLength=byte_length,
                byteOffset=byte_offset,
                target=target.value,
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
        )

    def build(self) -> GLTF:
        model = self.build_model()
        return GLTF(model=model, resources=self.file_resources)

    def build_and_export(self, filepath):
        self.build().export(filepath)
