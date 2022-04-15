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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from keras.optimizer_v2 import optimizer_v2 as keras_optimizer_v2
    from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as tf_optimizer_v2
    from tensorflow.keras.optimizers import Adam

    if issubclass(Adam, keras_optimizer_v2.OptimizerV2):
        del keras_optimizer_v2, tf_optimizer_v2, Adam
        from keras.optimizer_v2 import optimizer_v2
    elif issubclass(Adam, tf_optimizer_v2.OptimizerV2):
        del keras_optimizer_v2, tf_optimizer_v2, Adam
        from tensorflow.python.keras.optimizer_v2 import optimizer_v2
    else:
        raise TypeError("Cannot find Adam's base.")
except ModuleNotFoundError:
    from tensorflow.python.keras.optimizer_v2 import optimizer_v2