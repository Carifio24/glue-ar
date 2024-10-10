from numpy import array, zeros
from stl import Mesh
from typing import Iterable, List


class STLBuilder:

    def __init__(self):
        self.meshes: List[Mesh] = []

    def add_mesh(self,
                 vertices: List[Iterable[float]],
                 triangles: List[Iterable[int]]) -> STLBuilder:

        # Adapted from example at https://pypi.org/project/numpy-stl/
        verts_array = array(vertices)
        tris_array = array(triangles)
        mesh = Mesh(zeros(tris_array.shape[0], dtype=Mesh.dtype))
        for i, f in enumerate(tris_array):
            for j in range(3):
                mesh.vectors[i][j] = verts_array[f[j],:]

        self.meshes.append(mesh)
        return self

    def build(self) -> Mesh:
        return Mesh([mesh.data for mesh in self.meshes])

    def build_and_export(self, filepath: str):
        mesh = self.build()
        mesh.save(filepath)
