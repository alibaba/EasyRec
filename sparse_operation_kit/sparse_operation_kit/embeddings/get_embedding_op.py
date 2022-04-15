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

"""
These APIs are used along with TensorFlow 1.x.
They are similar to tf.get_variable()
"""

from tensorflow.python.framework import ops
import threading

_SparseOperationKitEmbeddingLayerStoreKey = "SparseOperationKitEmbeddingLayerStore"

class _EmbeddingLayerStore(threading.local):
    def __init__(self):
        super(_EmbeddingLayerStore, self).__init__()
        self._embedding_layer_container = dict()

    def _create_embedding(self, name, constructor, **kwargs):
        if constructor is None:
            raise ValueError("embedding_layer: '{}' does not exist and "
                             "cannot create it with constructor: "
                             "{}".format(name, constructor))
        embedding_layer = constructor(**kwargs)
        self._embedding_layer_container[name] = embedding_layer
        return embedding_layer

    def get_embedding(self, name, constructor=None, **kwargs):
        emb = self._embedding_layer_container.get(name, None)
        return emb or self._create_embedding(name, 
                        constructor=constructor, **kwargs)


def _get_embedding_store():
    emb_store = ops.get_collection(_SparseOperationKitEmbeddingLayerStoreKey)
    if not emb_store:
        emb_store = _EmbeddingLayerStore()
        ops.add_to_collection(_SparseOperationKitEmbeddingLayerStoreKey, emb_store)
    else:
        emb_store = emb_store[0]

    return emb_store


def get_embedding(name, constructor=None, **kwargs):
    """
    This method is used to get or create a embedding layer.
    
    Parameters
    ----------
    name: string
        unique name used to identify embedding layer.
    constructor: SOK.embedding_layer
        the construction function used to create a new embedding layer.
        When creating a new embedding layer, constructor(**kwargs) will be called.
    kwargs: keyword arguments for new embedding layer creation.

    Returns
    -------
    embedding_layer: embedding layer
        created by constructor(**kwargs)

    Examples
    --------
    .. code-block:: python

        # here to create a new embedding layer.
        emb_layer = sok.get_embedding(name="Emb", sok.All2AllDenseEmbedding,
                                      max_vocabulary_size_per_gpu=1024,
                                      embedding_vec_size=16,
                                      slot_num=10,
                                      nnz_per_slot=1,
                                      dynamic_input=False)
        outputs = emb_layer(inputs)
        ...
        # here to reuse already created embedding layer.
        emb_layer = sok.get_embedding(name="Emb")
        outputs_1 = emb_layer(inputs)
        ...
    """
    return _get_embedding_store().get_embedding(name=name, 
                constructor=constructor, **kwargs)