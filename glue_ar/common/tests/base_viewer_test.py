from os import remove
from random import randint, random, seed
from typing import Dict, Tuple, Type, Union

from glue.core import Data
from glue.core.state_objects import State
from glue.viewers.common.viewer import Viewer
from glue_jupyter import JupyterApplication
from glue_jupyter.ipyvolume.scatter import IpyvolumeScatterView
from glue_qt.app import GlueApplication
from glue_vispy_viewers.scatter.qt.scatter_viewer import VispyScatterViewer


class BaseViewerTest:

    def teardown_method(self, method):
        if getattr(self, "tmpfile", None) is not None:
            self.tmpfile.close()
            remove(self.tmpfile.name)
        if hasattr(self, 'viewer'):
            if hasattr(self.viewer, "close"):
                self.viewer.close(warn=False)
            self.viewer = None
        if hasattr(self, 'app'):
            if hasattr(self.app, 'close'):
                self.app.close()
        self.app = None

    def _create_application(self, app_type: str) -> Union[GlueApplication, JupyterApplication]:
        if app_type == "qt":
            return GlueApplication()
        elif app_type == "jupyter":
            return JupyterApplication()
        else:
            raise ValueError("Application type should be either qt or jupyter")

    def _viewer_class(self, viewer_type: str) -> Union[Type[VispyScatterViewer], Type[IpyvolumeScatterView]]:
        if viewer_type == "vispy":
            return VispyScatterViewer
        elif viewer_type == "ipyvolume":
            return IpyvolumeScatterView
        else:
            raise ValueError("Viewer type should be either vispy or ipyvolume")

    def _basic_state_dictionary(self, viewer_type: str) -> Dict[str, Tuple[str, State]]:
        raise NotImplementedError()

