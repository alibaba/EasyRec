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

from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import os, sys
import subprocess
import shutil

def _GetSOKVersion():
    _version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "sparse_operation_kit/core/")
    sys.path.append(_version_path)
    from _version import __version__
    version = __version__
    del __version__
    sys.path.pop(-1)
    return version

class SOKExtension(Extension):
    def __init__(self, name, cmake_file_path="./", sources=[], **kwargs):
        Extension.__init__(self, name, sources=sources, **kwargs)
        self._CMakeLists_dir = os.path.abspath(cmake_file_path)
        self._sok_lib_name = "libsparse_operation_kit.so"
        self._sok_unit_test_name = "libsok_unit_test.so"
        self._sok_compat_ops_name = "libsparse_operation_kit_compat_ops.so"

    @property
    def sok_lib_name(self):
        return self._sok_lib_name

    @property
    def sok_unit_test_name(self):
        return self._sok_unit_test_name

    @property
    def sok_compat_ops_name(self):
        return self._sok_compat_ops_name

class SOKBuildExtension(build_ext):
    def build_extensions(self):
        if os.getenv("SOK_NO_COMPILE") == "1":
            # skip compile the source codes
            return
        
        gpu_capabilities = ["70", "75", "80"]
        if os.getenv("SOK_COMPILE_GPU_SM"):
            gpu_capabilities = os.getenv("SOK_COMPILE_GPU_SM")
            gpu_capabilities = str(gpu_capabilities).strip().split(";")

        use_nvtx = "OFF"
        if os.getenv("SOK_COMPILE_USE_NVTX"):
            use_nvtx = "ON" if os.getenv("SOK_COMPILE_USE_NVTX") in ["1", "ON", "On", "on"] else "OFF"

        dedicated_cuda_stream = "ON"
        if os.getenv("SOK_COMPILE_ASYNC"):
            dedicated_cuda_stream = "OFF" if os.getenv("SOK_COMPILE_ASYNC") in ["0", "OFF", "Off", "off"] else "ON"

        unit_test = "OFF"
        if os.getenv("SOK_COMPILE_UNIT_TEST"):
            unit_test = "ON" if os.getenv("SOK_COMPILE_UNIT_TEST") in ["1", "ON", "On", "on"] else "OFF"

        cmake_build_type = "Release"
        if os.getenv("SOK_COMPILE_BUILD_TYPE"):
            cmake_build_type = "Debug" if os.getenv("SOK_COMPILE_BUILD_TYPE") in ["DEBUG", "debug", "Debug"] else "Release"

        make_jobs_num = "$(nproc)"
        if os.getenv("SOK_COMPILE_JOBS_NUM"):
            make_jobs_num = str(os.getenv("SOK_COMPILE_JOBS_NUM"))


        cmake_args = ["-DSM='{}'".format(";".join(gpu_capabilities)),
                      "-DUSE_NVTX={}".format(use_nvtx),
                      "-DSOK_ASYNC={}".format(dedicated_cuda_stream),
                      "-DSOK_UNIT_TEST={}".format(unit_test),
                      "-DCMAKE_BUILD_TYPE={}".format(cmake_build_type)]
        cmake_args = " ".join(cmake_args)

        build_dir = self.get_ext_fullpath(self.extensions[0].name).replace(
                        self.get_ext_filename(self.extensions[0].name), "")
        build_dir = os.path.abspath(build_dir)
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        for extension in self.extensions:
            try:
                subprocess.check_call("cmake {} {} && make -j{}".format(cmake_args, 
                                        extension._CMakeLists_dir, make_jobs_num), 
                                    shell=True,
                                    cwd=build_dir)
            except OSError as error:
                raise RuntimeError("Compile SOK faild, due to {}".format(str(error)))

        if sys.argv[1].startswith("develop"):
            # remove unfound so
            self.extensions = [ext for ext in self.extensions
                                if os.path.exists(self.get_ext_fullpath(ext.name))]

        # move dynamic libs
        self.move_dynamic_libs()

        # remove intermediate files
        self.clean_intermediates()

    def get_outputs(self):
        outputs = list()
        for ext in self.extensions:
            outputs.append(os.path.join(self.build_lib, ext.sok_lib_name))
            outputs.append(os.path.join(self.build_lib, ext.sok_compat_ops_name))
        return outputs

    def clean_intermediates(self):
        clean_folders = ["CMakeFiles", "kit_cc_impl", "unit_test", "lib"]
        clean_files = ["cmake_install.cmake", "CMakeCache.txt", "Makefile"]

        build_dir = self.get_ext_fullpath(self.extensions[0].name).replace(
                        self.get_ext_filename(self.extensions[0].name), "")
        for folder in clean_folders:
            _folder = os.path.join(build_dir, folder)
            if os.path.exists(_folder):
                shutil.rmtree(_folder)
        for _file in clean_files:
            _file = os.path.join(build_dir, _file)
            if os.path.exists(_file):
                os.remove(_file)

    def move_dynamic_libs(self):
        build_dir = self.get_ext_fullpath(self.extensions[0].name).replace(
                        self.get_ext_filename(self.extensions[0].name), "")
        lib_dir = os.path.join(build_dir, "lib")
        dest_dir = os.path.join(build_dir, self.extensions[0].name)
        dest_dir = os.path.join(dest_dir, "lib")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        lib_files = os.listdir(lib_dir)
        for filename in lib_files:
            shutil.move(os.path.join(lib_dir, filename),
                        os.path.join(dest_dir, filename))

    def copy_extensions_to_source(self):
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            build_dir = os.path.abspath(os.path.join(self.build_lib, "lib"))
            
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            modpath = fullname.split(".")
            package = ".".join(modpath[:-1])
            package_dir = build_py.get_package_dir(package)

            lib_name = ext.sok_lib_name
            compat_ops_name = ext.sok_compat_ops_name

            for filename in (lib_name, compat_ops_name):
                src_filename = os.path.join(build_dir, filename)
                dest_filename = os.path.join(package_dir, os.path.basename(filename))
                
                # Always copy, even if source is older than destination, to ensure
                # that the right extensions for the current Python/platform are used.
                from setuptools._distutils.file_util import copy_file
                copy_file(src_filename, dest_filename, verbose=self.verbose,
                          dry_run=self.dry_run)

            if ext._needs_stub:
                self.write_stub(package_dir or os.curdir, ext, True)

setup(
    name="merlin-sok",
    version=_GetSOKVersion(),
    author="NVIDIA",
    author_email="hugectr-dev@exchange.nvidia.com",
    url="https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/sparse_operation_kit",
    description="SparseOperationKit (SOK) is a python package wrapped GPU accelerated"
                 " operations dedicated for sparse training / inference cases.",
    long_description="SparseOperationKit (SOK) is a python package wrapped GPU accelerated "
                     "operations dedicated for sparse training / inference cases. "
                     "It is designed to be compatible with common DeepLearning (DL) frameworks, "
                     "for instance, TensorFlow. "
                     "Most of the algorithm implementations in SOK are extracted from HugeCTR, "
                     "which is a GPU-accelerated recommender framework designed to distribute "
                     "training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). "
                     "If you are looking for a very efficient solution for CTRs, please check HugeCTR.",
    extras_require={"tensorflow": "tensorflow>=1.15"},
    license="Apache 2.0",
    platforms=["Linux"],
    python_requires='>=3', # TODO: make it compatible with python2.7
    packages=find_packages(),
    package_dir={"": "./"},
    cmdclass={"build_ext": SOKBuildExtension},
    ext_modules=[SOKExtension("sparse_operation_kit")],
)
