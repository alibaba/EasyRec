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

import tensorflow as tf

import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../")))

import sparse_operation_kit as sok

class SingleWorkerbase(object):
    def __init__(self, **kwargs):
        print("[INFO]: single worker testing.")
        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            init_re = sok.Init(**kwargs)

    def call(self):
        raise RuntimeError("The unit testing should be implemented inside this function.")

    def __call__(self, *args, **kwargs):
        re = self.call()
        print("[INFO] %s passed." %self.__class__.__name__)
        return re

if __name__ == "__main__":
    obj = SingleWorkerbase()
    obj()