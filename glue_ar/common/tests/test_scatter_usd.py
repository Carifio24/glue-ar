from sys import platform
from tempfile import NamedTemporaryFile

from pxr import Usd
import pytest

from glue_ar.common.export import export_viewer
from glue_ar.common.shapes import sphere_points_count, sphere_triangles_count
from glue_ar.common.tests.helpers import APP_VIEWER_OPTIONS
from glue_ar.common.tests.test_scatter import BaseScatterTest
from glue_ar.usd_utils import material_for_mesh
from glue_ar.utils import export_label_for_layer, hex_to_components, iterator_count, layers_to_export, xyz_bounds


class TestVispyScatterUSD(BaseScatterTest):

    @pytest.mark.parametrize("app_type,viewer_type", APP_VIEWER_OPTIONS)
    def test_basic_export(self, app_type: str, viewer_type: str):
        if app_type == "jupyter" and viewer_type == "vispy" and platform == "win32":
            return
        self.basic_setup(app_type, viewer_type)
        bounds = xyz_bounds(self.viewer.state, with_resolution=False)
        self.tmpfile = NamedTemporaryFile(suffix=".usdc", delete=False)
        self.tmpfile.close()
        layer_states = [layer.state for layer in layers_to_export(self.viewer)]
        export_viewer(self.viewer.state,
                      states=layer_states,
                      bounds=bounds,
                      state_dictionary=self.state_dictionary,
                      filepath=self.tmpfile.name,
                      compression=None)

        stage = Usd.Stage.Open(self.tmpfile.name)
        world = stage.GetDefaultPrim()
        assert str(world.GetPath()) == "/world"

        layer = self.viewer.layers[0]
        label = export_label_for_layer(layer.state)
        identifier = label.replace(" ", "_")
        _, options = self.state_dictionary[label]

        # The default ipyvolume geometry type is diamond
        theta_resolution: int = getattr(options, "resolution", 3)
        phi_resolution: int = getattr(options, "resolution", 3)
        sphere_pts_count = sphere_points_count(theta_resolution=theta_resolution, phi_resolution=phi_resolution)
        sphere_tris_count = sphere_triangles_count(theta_resolution=theta_resolution, phi_resolution=phi_resolution)
        expected_vert_cts = [3] * sphere_tris_count * self.n

        color_precision = 5
        color_components = [round(c / 255, color_precision) for c in hex_to_components(layer.state.color)]

        # There should be 6 total prims:
        # The xform and the mesh (single color means one mesh)
        # The top-level world prim
        # The light
        # The material
        # The PBR shader
        assert iterator_count(stage.TraverseAll()) == 6

        point_mesh = stage.GetPrimAtPath(f"/world/xform_{identifier}_0/mesh_{identifier}_0")
        assert point_mesh is not None
        points = list(point_mesh.GetAttribute("points").Get())
        assert len(points) == sphere_pts_count * self.n
        vertex_counts = list(point_mesh.GetAttribute("faceVertexCounts").Get())
        assert vertex_counts == expected_vert_cts
        vertex_indices = list(point_mesh.GetAttribute("faceVertexIndices").Get())
        assert len(vertex_indices) == sphere_tris_count * 3 * self.n

        material = material_for_mesh(point_mesh)
        pbr_shader = stage.GetPrimAtPath(f"{material.GetPath()}/PBRShader")
        assert pbr_shader.GetAttribute("inputs:metallic").Get() == 0.0
        assert pbr_shader.GetAttribute("inputs:roughness").Get() == 1.0
        assert round(pbr_shader.GetAttribute("inputs:opacity").Get(), color_precision) == \
               round(layer.state.alpha, color_precision)
        assert [round(c, color_precision) for c in pbr_shader.GetAttribute("inputs:diffuseColor").Get()] == \
               color_components
