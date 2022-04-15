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

from tensorflow.python.framework import ops

class HorovodStrategy(object):
    def __init__(self):
        import horovod.tensorflow as hvd
        self._hvd = hvd
    def scope(self):
        return ops.NullContextmanager()

    def run(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def reduce(self, combiner, tensors):
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            return [self._hvd.allreduce(tensor) for tensor in tensors]
        else:
            return self._hvd.allreduce(tensors)

class OneDeviceStrategy(object):
    def scope(self):
        return ops.NullContextmanager()
    
    def run(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def reduce(self, combiner, tensors):
        return tensors