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

import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../")))
import utility
import numpy as np

def generate_datas(args):
    counts = args.iter_num // 10

    total_samples, total_labels = None, None

    for _ in range(counts):
        random_samples, random_labels = utility.generate_random_samples(
                                            num_of_samples=args.global_batch_size * 10,
                                            vocabulary_size=args.vocabulary_size,
                                            slot_num=args.slot_num,
                                            max_nnz=args.nnz_per_slot,
                                            use_sparse_mask=False)
        if total_samples is None:
            total_samples = random_samples
            total_labels = random_labels
        else:
            total_samples = np.concatenate([total_samples, random_samples], axis=0)
            total_labels = np.concatenate([total_labels, random_labels], axis=0)

    utility.save_to_file(args.filename, total_samples, total_labels)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate dataset for DenseDemo")

    parser.add_argument("--global_batch_size", type=int, 
                        required=True, 
                        help="the mini-batchsize used in each iteration.")
    parser.add_argument("--slot_num", type=int,
                        help="the number of feature fields",
                        required=True)
    parser.add_argument("--nnz_per_slot", type=int,
                        help="the number of keys in each slot",
                        required=True)
    parser.add_argument("--vocabulary_size", type=int,
                        required=False, default=1024 * 8)
    parser.add_argument("--iter_num", type=int,
                        help="the number of training iterations",
                        required=True)
    parser.add_argument("--filename", type=str,
                        help="the filename used to save the generated datas.",
                        required=False, default=r"./data.file")

    args = parser.parse_args()

    generate_datas(args)

