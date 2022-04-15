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
from typing import List
import tensorflow as tf

class CriteoTsvReader:
    """
    Input reader for pre-processed Criteo data.

    Raw Criteo data is assumed to be preprocessed in the following way:
    1. Missing values are replaced with zeros.
    2. Negative values are replaced with zeros.
    3. Integer features are transformed by log(x+1) and are hence tf.float32.
    4. Categorical data is bucketized and are hence tf.int32
    """
    def __init__(self,
                 file_pattern: str,
                 num_dense_features: int,
                 vocab_sizes: List[int],
                 batch_size: int,
                 sharding: bool = False):
        self._file_pattern = file_pattern
        self._num_dense_features = num_dense_features
        self._vocab_sizes = vocab_sizes
        self._batch_size = batch_size
        self._sharding = sharding

    def __call__(self, input_ctx: tf.distribute.InputContext = None) -> tf.data.Dataset:
        batch_size = input_ctx.get_per_replica_batch_size(self._batch_size) if input_ctx else self._batch_size

        @tf.function
        def _parse_fn(example: tf.Tensor):
            label_defaults = [[0]]
            dense_defaults = [
                [0.0] for _ in range(self._num_dense_features)]
            num_sparse_features = len(self._vocab_sizes)
            categorical_defaults = [
                [0] for _ in range(num_sparse_features)]
            record_defaults = label_defaults + dense_defaults + categorical_defaults
            fields = tf.io.decode_csv(example, record_defaults, field_delim="\t", na_value="-1")

            num_labels = 1
            label = tf.reshape(fields[0], [batch_size, 1])

            features = {}
            num_dense = len(dense_defaults)

            dense_features = []
            offset = num_labels
            for idx in range(num_dense):
                dense_features.append(fields[idx + offset])
            features["dense_features"] = tf.stack(dense_features, axis=1)

            offset += num_dense
            features["sparse_features"] = {}

            for idx in range(num_sparse_features):
                features["sparse_features"][str(idx)] = fields[idx + offset]

            return features, label
        
        filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)
        if self._sharding and input_ctx and input_ctx.num_input_pipelines > 1:
            filenames = filenames.shard(input_ctx.num_input_pipelines, 
                                        input_ctx.input_pipeline_id)

        num_shards_per_host = 1
        if self._sharding:
            num_shards_per_host = 16
        
        def make_dataset(shard_index):
            filenames_for_shard = filenames.shard(num_shards_per_host, shard_index)
            dataset = tf.data.TextLineDataset(filenames_for_shard)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.map(_parse_fn, 
                                  num_parallel_calls=tf.data.AUTOTUNE)
            return dataset

        indices = tf.data.Dataset.range(num_shards_per_host)
        dataset = indices.interleave(
            map_func=make_dataset,
            num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    dataset = CriteoTsvReader(file_pattern=r"./train/*",
                              num_dense_features=13,
                              vocab_sizes=[39884407, 39043, 17289, 7420, 20263, 
                                            3, 7120, 1543, 63, 38532952, 2953546, 
                                            403346, 10, 2208, 11938, 155, 4, 976, 
                                            14, 39979772, 25641295, 39664985, 585935, 
                                            12972, 108, 36],
                              batch_size=16384,
                              sharding=False)()

    for step, (features, labels) in enumerate(dataset):
        # print(features)
        print(labels)

    
