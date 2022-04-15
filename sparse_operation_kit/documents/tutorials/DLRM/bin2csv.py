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
import os
import struct
import csv
import multiprocessing

def get_file_size_in_bytes(filename):
    return os.path.getsize(filename)

class BinaryToCSV(object):
    def __init__(self, args, save_header=False):
        self.args = args
        self.save_header = save_header

        self.slot_size_array = [39884407, 39043, 17289, 7420, 20263, 
                       3, 7120, 1543, 63, 38532952, 2953546, 
                       403346, 10, 2208, 11938, 155, 4, 976, 
                       14, 39979772, 25641295, 39664985, 585935, 
                       12972, 108, 36]
        self.num_dense_features = 13
        self.num_cate_features = 26
        self.item_num_per_sample = 1 + self.num_dense_features + self.num_cate_features
        self.sample_format = r"1I" + str(self.num_dense_features) + "f" +\
                             str(self.num_cate_features) + "I"
        self.dense_feature_keys = [
            f"int-feature-{x + 1}" for x in range(self.num_dense_features)]
        self.cate_feature_keys = [
            "categorical-feature-%d" % x for x in range(self.num_dense_features + 1, 40)]
        self.label_key = "clicked"
        self.header = [self.label_key] + self.dense_feature_keys + self.cate_feature_keys

        self.sample_size_in_bytes = 1 * 4 + self.num_dense_features * 4 +\
                                    self.num_cate_features * 4
        self.file_size_in_bytes = get_file_size_in_bytes(self.args.input_file)
        if self.file_size_in_bytes % self.sample_size_in_bytes != 0:
            raise RuntimeError("The filesize of {} is not divisible to samplesize.".format(
                self.args.input_file))

        self.samples_num = self.file_size_in_bytes // self.sample_size_in_bytes
        self.samples_num_each_shard = self.samples_num // self.args.num_output_files

        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)

    def __call__(self):
        if 1 == self.args.num_output_files:
            self.convert_func(shard_id=0)
        else:
            processes = os.cpu_count() // 2 if os.cpu_count() // 2 >= 1 else 1
            chunksize = self.args.num_output_files // processes
            with multiprocessing.Pool(processes=processes) as pool:
                pool.imap(self.convert_func, range(self.args.num_output_files), 
                          chunksize=chunksize)
                pool.close()
                pool.join()

    def convert_func(self, shard_id):
        my_begin_index = self.samples_num_each_shard * shard_id
        my_end_index = self.samples_num_each_shard * (shard_id + 1)
        my_end_index = my_end_index if my_end_index <= self.samples_num else self.samples_num

        save_name = os.path.join(self.args.output_path, 
                                 self.args.save_prefix + str(shard_id) + ".csv")

        with open(self.args.input_file, "rb") as InFile,\
            open(save_name, "w") as OutFile:
            # skip samples not belonging to me
            InFile.seek(self.sample_size_in_bytes * my_begin_index, 0)
            # Read my samples
            data_buffer = InFile.read((my_end_index - my_begin_index) * self.sample_size_in_bytes)
            # convert to numerical data
            unpack_format = self.sample_format * (my_end_index - my_begin_index)
            data = struct.unpack(unpack_format, data_buffer)
            data = [data[i * self.item_num_per_sample : (i + 1) * self.item_num_per_sample] 
                    for i in range(my_end_index - my_begin_index)]
            # save to CSV file.
            writer = csv.writer(OutFile, delimiter="\t")
            if self.save_header:
                writer.writerow(self.header) # TF.dataset cannot correctly skip header
            writer.writerows(data)
        print("[INFO]: Saved %s done." %(save_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True,
                        help="the filename of the binary file")
    parser.add_argument("--num_output_files", type=int, required=False,
                        default=1, help="the number of shards")
    parser.add_argument("--output_path", type=str, required=False,
                        default="./", help="the directory to save files.")
    parser.add_argument("--save_prefix", type=str, required=True,
                        help="the prefix for saving outptu shards")

    args = parser.parse_args()

    BinaryToCSV(args)()