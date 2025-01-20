from sys import platform
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from stl import Mesh

from glue_ar.common.export import export_viewer
from glue_ar.common.shapes import sphere_points_count, sphere_triangles_count
from glue_ar.common.tests.helpers import APP_VIEWER_OPTIONS
from glue_ar.common.tests.test_scatter import BaseScatterTest
from glue_ar.utils import export_label_for_layer, layers_to_export, mask_for_bounds, xyz_bounds, xyz_for_layer


class TestScatterSTL(BaseScatterTest):

    @pytest.mark.parametrize("app_type,viewer_type", APP_VIEWER_OPTIONS)
    def test_basic_export(self, app_type: str, viewer_type: str):
        if app_type == "jupyter" and viewer_type == "vispy" and platform == "win32":
            return
        self.basic_setup(app_type, viewer_type)
        bounds = xyz_bounds(self.viewer.state, with_resolution=False)
        self.tmpfile = NamedTemporaryFile(suffix=".stl", delete=False)
        self.tmpfile.close()
        layer_states = [layer.state for layer in layers_to_export(self.viewer)]
        export_viewer(self.viewer.state,
                      states=layer_states,
                      bounds=bounds,
                      state_dictionary=self.state_dictionary,
                      filepath=self.tmpfile.name,
                      compression=None)

        stl = Mesh.from_file(self.tmpfile.name)

        layer = self.viewer.layers[0]
        mask = mask_for_bounds(self.viewer.state, layer.state, bounds)
        data = xyz_for_layer(self.viewer.state, layer.state,
                             preserve_aspect=self.viewer.state.native_aspect,
                             mask=mask,
                             scaled=True)

        data = data[:, [1, 2, 0]]

        # Check that the center of each sphere mesh matches the
        # corresponding data point
        tolerance = 1e-6
        layer_label = export_label_for_layer(layer)
        method, options = self.state_dictionary[layer_label]
        assert method == "Scatter"

        # TODO: 3 is the value for ipyvolume's diamond, which is the ipv default
        # But we should make this more robust
        theta_resolution: int = getattr(options, "resolution", 3)
        phi_resolution: int = getattr(options, "resolution", 3)
        points_count = sphere_points_count(theta_resolution, phi_resolution)
        triangle_count = sphere_triangles_count(theta_resolution, phi_resolution)
        for index in range(self.data1.size):
            lower_vector_index = index * triangle_count
            upper_vector_index = lower_vector_index + triangle_count
            vectors = stl.vectors[lower_vector_index:upper_vector_index]
            vectors = vectors.reshape(vectors.shape[0] * vectors.shape[1], vectors.shape[2])
            points = np.unique(vectors, axis=0)
            data_point = data[index]
            center = sum(points) / points_count
            assert all(abs(center[i] - data_point[i]) < tolerance for i in range(center.shape[0]))
