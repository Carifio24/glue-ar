from glue.core import Data
from numpy import arange, ones
import pytest


@pytest.fixture
def volume_data() -> Data:
    return Data(label='d1',
                x=arange(24).reshape((2, 3, 4)),
                y=ones((2, 3, 4)),
                z=arange(100, 124).reshape((2, 3, 4)))
