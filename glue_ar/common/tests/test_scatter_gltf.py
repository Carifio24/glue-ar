from sys import platform
from tempfile import NamedTemporaryFile

from gltflib import AccessorType, AlphaMode, BufferTarget, ComponentType, GLTFModel
from gltflib.gltf import GLTF
import pytest

from glue_ar.common.export import export_viewer
from glue_ar.common.shapes import sphere_points_count, sphere_triangles, sphere_triangles_count
from glue_ar.common.tests.gltf_helpers import count_indices, count_vertices, unpack_vertices
from glue_ar.common.tests.helpers import APP_VIEWER_OPTIONS
from glue_ar.common.tests.test_scatter import BaseScatterTest
from glue_ar.gltf_utils import index_export_option
from glue_ar.utils import export_label_for_layer, hex_to_components, layers_to_export, mask_for_bounds, \
                          xyz_bounds, xyz_for_layer


class TestScatterGLTF(BaseScatterTest):

    # TODO: How can we test the properties of compressed files?
    @pytest.mark.parametrize("app_type,viewer_type", APP_VIEWER_OPTIONS)
    def test_basic_export(self, app_type: str, viewer_type: str):
        if app_type == "jupyter" and viewer_type == "vispy" and platform == "win32":
            return
        self.basic_setup(app_type, viewer_type)
        bounds = xyz_bounds(self.viewer.state, with_resolution=False)
        self.tmpfile = NamedTemporaryFile(suffix=".gltf", delete=False)
        self.tmpfile.close()
        layer_states = [layer.state for layer in layers_to_export(self.viewer)]
        export_viewer(self.viewer.state,
                      layer_states=layer_states,
                      bounds=bounds,
                      state_dictionary=self.state_dictionary,
                      filepath=self.tmpfile.name,
                      compression=None)

        gltf: GLTF = GLTF.load(self.tmpfile.name)
        model = gltf.model
        assert isinstance(model, GLTFModel)

        assert model.buffers is not None and len(model.buffers) == 1
        assert model.bufferViews is not None and len(model.bufferViews) == 1 + self.n
        assert model.accessors is not None and len(model.accessors) == 1 + self.n
        assert model.meshes is not None and len(model.meshes) == self.n
        assert model.nodes is not None and len(model.nodes) == self.n
        assert model.materials is not None and len(model.materials) == 1

        material = model.materials[0]
        layer = self.viewer.layers[0]
        label = export_label_for_layer(layer)
        method, options = self.state_dictionary[label]
        assert method == "Scatter"
        export_state_cls = self._export_state_class(self.viewer_type)
        assert isinstance(options, export_state_cls)
        color_components = [c / 256 for c in hex_to_components("#fedcba")] + [layer.state.alpha]
        assert material.alphaMode == AlphaMode.BLEND.value
        assert material.pbrMetallicRoughness is not None
        assert material.pbrMetallicRoughness.baseColorFactor == color_components
        assert material.pbrMetallicRoughness.roughnessFactor == 1
        assert material.pbrMetallicRoughness.metallicFactor == 0

        assert all(mesh.primitives[0].indices == 0 for mesh in model.meshes)

        # TODO: 3 is the value for ipyvolume's diamond, which is the ipv default
        # But we should make this more robust
        theta_resolution: int = getattr(options, "resolution", 3)
        phi_resolution: int = getattr(options, "resolution", 3)
        triangles_count = sphere_triangles_count(theta_resolution=theta_resolution,
                                                 phi_resolution=phi_resolution)
        points_count = sphere_points_count(theta_resolution=theta_resolution,
                                           phi_resolution=phi_resolution)
        index_format = index_export_option(points_count)
        assert count_indices(gltf,
                             model.buffers[0],
                             model.bufferViews[0],
                             export_option=index_format) == triangles_count
        assert count_vertices(gltf, model.buffers[0], model.bufferViews[1]) == points_count

        assert model.bufferViews[0].target == BufferTarget.ELEMENT_ARRAY_BUFFER.value
        assert model.bufferViews[1].target == BufferTarget.ARRAY_BUFFER.value

        indices_accessor = model.accessors[0]
        assert indices_accessor.bufferView == 0
        expected_indices_type = index_format.component_type
        assert indices_accessor.componentType == expected_indices_type.value
        assert indices_accessor.count == triangles_count * 3
        assert indices_accessor.type == AccessorType.SCALAR.value
        assert indices_accessor.min == [0]
        assert indices_accessor.max == [max(max(tri) for tri in sphere_triangles(theta_resolution=theta_resolution,
                                                                                 phi_resolution=phi_resolution))]

        for index, accessor in enumerate(model.accessors[1:]):
            assert accessor.bufferView == index + 1
            assert accessor.componentType == ComponentType.FLOAT.value
            assert accessor.count == points_count
            assert accessor.type == AccessorType.VEC3.value

        mask = mask_for_bounds(self.viewer.state, layer.state, bounds)
        data = xyz_for_layer(self.viewer.state, layer.state,
                             preserve_aspect=self.viewer.state.native_aspect,
                             mask=mask,
                             scaled=True)
        data = data[:, [1, 2, 0]]

        # Check that the center of each sphere mesh matches the
        # corresponding data point
        tolerance = 1e-7
        for index, mesh in enumerate(model.meshes):
            buffer_view = model.bufferViews[mesh.primitives[0].attributes.POSITION]
            points = unpack_vertices(gltf, model.buffers[0], buffer_view)
            n_points = len(points)
            center = tuple(sum(p[i] for p in points) / n_points for i in range(3))
            data_point = data[index]
            assert all(abs(center[i] - data_point[i]) < tolerance for i in range(len(center)))
