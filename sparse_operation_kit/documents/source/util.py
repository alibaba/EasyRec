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
import base64

def convert_pic_to_base64(picname):
    with open(picname, "rb") as pic:
        ls_f = base64.b64encode(pic.read())
        print(ls_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--picname", "-p", type=str,
                        help="the picture name",
                        required=True)

    args = parser.parse_args()

    convert_pic_to_base64(args.picname)