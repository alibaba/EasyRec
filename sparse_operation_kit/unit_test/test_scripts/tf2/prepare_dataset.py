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
                    os.path.dirname(os.path.abspath(__file__)), 
                    "../../../documents/tutorials/DenseDemo/")))

from gen_data import generate_datas
from split_data import split_data

from importlib.util import find_spec

has_horovod = find_spec("horovod")
if has_horovod is None:
    os.system("HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod && horovodrun --check-build")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate dataset")

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
                        required=False, default=r"./datas.file")
    parser.add_argument("--split_num", type=int,
                        required=True,
                        help="the number of shards to be splited.")
    parser.add_argument("--save_prefix", type=str,
                        required=True,
                        help="the prefix used to save splits.")

    args = parser.parse_args()

    generate_datas(args)
    split_data(args.filename, args.split_num, args.save_prefix)