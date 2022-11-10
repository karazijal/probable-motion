from . import log
from . import convert
# Do not reorder this
from . import flow_reconstruction
from . import data
from . import environment
from . import postproc
from . import random_state
from . import schedules
from . import visualisation

import torch
import numpy as np
from typing import Iterable

def get_shape(t):
    if torch.is_tensor(t) or type(t).__module__ == np.__name__:
        return t.shape
    if isinstance(t, (Iterable)):
        return len(t)
    else:
        return t
