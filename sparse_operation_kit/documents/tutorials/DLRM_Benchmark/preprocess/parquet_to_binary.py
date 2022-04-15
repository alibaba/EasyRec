# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import glob
import argparse
import tqdm
import subprocess
import random
import time


def process_file(f, dst):
    label_cols = [['label'], np.float32, 'label/']
    dense_cols = [['I' + str(i) for i in range(1, 14)], np.float32, 'dense/']
    category_cols = [['C' + str(i) for i in range(1, 27)], np.int32, 'category/']

    data = pd.read_parquet(f)

    for cols, dtype, sub_dir in [label_cols, dense_cols, category_cols]:
        sub_data = data[cols].astype(dtype)
        sub_data = sub_data.to_records(index=False)
        sub_data = sub_data.tobytes()
        dst_file = os.path.join(dst, sub_dir, f.split('/')[-1] + '.bin')
        with open(dst_file, 'wb') as dst_fd:
            dst_fd.write(sub_data)


def get_categorical_feature_type(size: int):
    types = (np.int8, np.int16, np.int32)

    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(f'Categorical feature of size {size} is too big for defined types')


def process_file_split_category(f, dst, vocab_sizes):
    label_cols = (['label'], np.bool_, 'label/')
    dense_cols = (['I' + str(i) for i in range(1, 14)], np.float16, 'dense/')
    all_cols = [label_cols, dense_cols]
    for i in range(26):
        dtype = get_categorical_feature_type(vocab_sizes[i])
        target_dir = 'category/%d/'%i
        category_col = (['C' + str(i + 1)], dtype, target_dir)
        all_cols.append(category_col)

    data = pd.read_parquet(f)

    for cols, dtype, sub_dir in all_cols:
        sub_data = data[cols].astype(dtype)
        sub_data = sub_data.to_records(index=False)
        sub_data = sub_data.tobytes()
        dst_file = os.path.join(dst, sub_dir, f.split('/')[-1] + '.bin')
        with open(dst_file, 'wb') as dst_fd:
            dst_fd.write(sub_data)


def run(src_dir, mid_dir, dst_dir, parallel_jobs=40, shuffle=False, split_category=False, workflow_dir=None):
    start_time = time.time()

    print('Processing files ...')
    src_files = glob.glob(os.path.join(src_dir, '*.parquet'))
    if shuffle:
        random.shuffle(src_files)
    os.system('rm -rf %s'%mid_dir)
    for sub_dir in ['label', 'dense', 'category']:
        os.makedirs(os.path.join(mid_dir, sub_dir))
    if split_category:
        for i in range(26):
            os.makedirs(os.path.join(mid_dir, 'category/%d/'%i))

        assert(workflow_dir is not None)
        import nvtabular as nvt
        workflow = nvt.Workflow.load(workflow_dir)
        vocab_sizes = []
        embedding_sizes = nvt.ops.get_embedding_sizes(workflow)
        for key in ['C' + str(x) for x in range(1, 27)]:
            vocab_sizes.append(embedding_sizes[key][0])

    if not split_category:
        Parallel(n_jobs=parallel_jobs)(delayed(process_file)(f, mid_dir) for f in tqdm.tqdm(src_files))
    else:
        Parallel(n_jobs=parallel_jobs)(delayed(process_file_split_category)(f, mid_dir, vocab_sizes) for f in tqdm.tqdm(src_files))

    print('Files conversion done.')

    os.system('rm -rf %s'%dst_dir)
    os.makedirs(dst_dir)

    print('Concatenating files ...')
    for sub_dir in ['label', 'dense', 'category']:
        mid_sub_dir = os.path.join(mid_dir, sub_dir)
        if split_category and sub_dir == 'category':
            for i in range(26):
                sub_category_dir = os.path.join(mid_sub_dir, str(i))
                os.system(f'cat {sub_category_dir}/*.bin > {dst_dir}/{sub_dir}_{i}.bin')
        else:
            os.system(f'cat {mid_sub_dir}/*.bin > {dst_dir}/{sub_dir}.bin')

    print('Done, %.3f s.'%(time.time() - start_time))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str)
    parser.add_argument('mid_dir', type=str)
    parser.add_argument('dst_dir', type=str)
    parser.add_argument('--parallel_jobs', default=40, type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--split_category', default=False, type=bool)
    parser.add_argument('--workflow_dir', default=None, type=str)
    args = parser.parse_args()

    run(args.src_dir, args.mid_dir, args.dst_dir, args.parallel_jobs, args.shuffle, args.split_category, args.workflow_dir)
