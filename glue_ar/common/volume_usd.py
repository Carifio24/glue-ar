from glue_vispy_viewers.volume.viewer_state import Vispy3DVolumeViewerState
import neurovolume as nv

from glue_ar.common.export_options import ar_layer_export
from glue_vispy_viewers.volume.layer_state import VolumeLayerState

from glue_ar.common.usd_builder import USDBuilder
from glue_ar.common.volume_export_options import ARVoxelExportOptions
from glue_ar.utils import BoundsWithResolution, frb_for_layer, xyz_bounds


__all__ = ["add_volume_layer_usd"]


@ar_layer_export(VolumeLayerState, "Volume", ARVoxelExportOptions, ("usda", "usdc", "usdz"))
def add_volume_layer_usd(builder: USDBuilder,
                         viewer_state: Vispy3DVolumeViewerState,
                         layer_state: VolumeLayerState,
                         options: ARVoxelExportOptions,
                         bounds: Optional[BoundsWithResolution] = None):

    bounds = bounds or xyz_bounds(viewer_state, with_resolution=True)
    data = frb_for_layer(viewer_state, layer_state, bounds)

    transpose = (0, 1, 2)  # TODO: What should this be?
    nv.ndarray_to_vdb(
        nv.prep_ndarray(data, transpose),
        "volume",
        output_dir=".",
    )
    
    builder.add_volume("volume.vdb")
