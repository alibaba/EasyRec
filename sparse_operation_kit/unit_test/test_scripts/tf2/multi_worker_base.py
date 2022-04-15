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

import os, json
from multiprocessing import Process

class MultiWorkerbase(object):
    def __init__(self):
        print("[INFO]: multi worker testing.")

    def init(self, task_id):
        ip = "127.0.0.1"
        port = "12345"
        port1 = "12346"
        if 0 == task_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        elif 1 == task_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        else:
            raise RuntimeError("task_id can only be one of [0, 1].")

        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {"worker": [ip + ":" + port, ip + ":" + port1]},
            "task": {"type": "worker", "index": task_id}
        })

        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

        self.strategy = tf.distribute.MultiWorkerMirroredStrategy(resolver)
        with self.strategy.scope():
            init_re = sok.Init()

    def call(self, *args, **kwargs):
        raise RuntimeError("The unit testing should be implemented inside this function.")

    def worker_0(self):
        self.init(task_id=0)
        return self.call()

    def worker_1(self):
        self.init(task_id=1)
        return self.call()

    def __call__(self, *args, **kwargs):
        p0 = Process(target=self.worker_0, args=())
        p1 = Process(target=self.worker_1, args=())
        p0.start()
        p1.start()

        if p0.is_alive():
            p0.join()
        
        if p1.is_alive():
            p1.join()


if __name__ == "__main__":
    obj = MultiWorkerbase()
    obj()