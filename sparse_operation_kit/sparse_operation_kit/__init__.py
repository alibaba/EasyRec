"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

print("[INFO]: %s is imported" %__name__)

from sparse_operation_kit.core._version import __version__

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
# TODO: this env should be left to users
# os.environ["TF_GPU_THREAD_COUNT"] = "16"
del os

# # ---------- import submodule ----------- #
import sparse_operation_kit.tf

# ------------ import items into root package -------- #
from sparse_operation_kit.core.initialize import Init
from sparse_operation_kit.core.context_scope import OptimizerScope
from sparse_operation_kit.embeddings.distributed_embedding import DistributedEmbedding
from sparse_operation_kit.embeddings.all2all_dense_embedding import All2AllDenseEmbedding
from sparse_operation_kit.embeddings.get_embedding_op import get_embedding
from sparse_operation_kit.saver.Saver import Saver
from sparse_operation_kit.optimizers.utils import split_embedding_variable_from_others
from sparse_operation_kit.core.embedding_layer_handle import GraphKeys

__all__ = [item for item in dir() if not item.startswith("__")]