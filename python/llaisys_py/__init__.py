# 在任何会加载 libllaisys/CUDA 的 import 之前修正：CUDA_VISIBLE_DEVICES="" 会导致 cudaGetDeviceCount() 一直为 0
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from .runtime import RuntimeAPI
from .libllaisys import DeviceType
from .libllaisys import DataType
from .libllaisys import MemcpyKind
from .libllaisys import llaisysStream_t as Stream
from .tensor import Tensor
from .ops import Ops
from . import models
from .models import *

__all__ = [
    "RuntimeAPI",
    "DeviceType",
    "DataType",
    "MemcpyKind",
    "Stream",
    "Tensor",
    "Ops",
    "models",
]
