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

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "../../unit_test/test_scripts/tf2/")))
from utils import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "DenseDemo/")))
from models import SOKDenseDemo

def TFDataset(filename, batchsize, as_sparse_tensor, repeat=1):
    samples, labels = restore_from_file(filename)
    dataset = tf_dataset(keys=samples, labels=labels,
                         batchsize=batchsize,
                         to_sparse_tensor=as_sparse_tensor,
                         repeat=repeat)
    del samples
    del labels
    return dataset


def get_dataset(global_batch_size,
                read_batchsize,
                iter_num=10,
                vocabulary_size=1024,
                slot_num=10,
                max_nnz=5,
                use_sparse_mask=False,
                repeat=1):
    random_samples, ramdom_labels = generate_random_samples(
                                num_of_samples=global_batch_size * iter_num,
                                vocabulary_size=vocabulary_size,
                                slot_num=slot_num,
                                max_nnz=max_nnz,
                                use_sparse_mask=use_sparse_mask)
    dataset = tf_dataset(keys=random_samples, 
                        labels=ramdom_labels,
                        batchsize=read_batchsize,
                        to_sparse_tensor=use_sparse_mask,
                        repeat=repeat)
    return dataset