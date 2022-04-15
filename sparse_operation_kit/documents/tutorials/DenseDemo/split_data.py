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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "../")))
import utility
import numpy as np
from multiprocessing import Process

def split_data(filename, split_num, save_prefix):
    samples, labels = utility.restore_from_file(filename)

    if samples.shape[0] % split_num != 0:
        raise RuntimeError("the number of samples: %d is not divisible by split_num: %d"
                            %(samples.shape[0], split_num))

    def split_func(split_id):
        # -------- choose a whole part ------------------ #
        each_split_sample_num = samples.shape[0] // split_num
        my_samples = samples[split_id * each_split_sample_num: (split_id + 1) * each_split_sample_num]
        my_labels = labels[split_id * each_split_sample_num: (split_id + 1) * each_split_sample_num]

        utility.save_to_file(save_prefix + str(split_id) + ".file", my_samples, my_labels)

    pros = list()
    for i in range(split_num):
        p = Process(target=split_func, args=(i,))
        pros.append(p)
        p.start()

    for p in pros:
        if p.is_alive():
            p.join()
    
    print("[INFO]: split dataset finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="split dataset into multiple shards")

    parser.add_argument("--filename", type=str, 
                        required=True, 
                        help="the filename of the whole dataset")
    parser.add_argument("--split_num", type=int,
                        required=True,
                        help="the number of shards to be splited.")
    parser.add_argument("--save_prefix", type=str,
                        required=True,
                        help="the prefix used to save splits.")

    args = parser.parse_args()
    
    split_data(args.filename, args.split_num, args.save_prefix)