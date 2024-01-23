#!/usr/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_ABI=$(python -c 'import tensorflow as tf; print(str(tf.sysconfig.CXX11_ABI_FLAG if "CXX11_ABI_FLAG" in dir(tf.sysconfig) else 0))')
echo "tensorflow include path: $TF_INC"
echo "tensorflow link flags: $TF_LFLAGS"
echo "CXX11_ABI_FLAG=$TF_ABI"

script_path=`readlink -f $0`
ops_dir=`dirname $script_path`
ops_src_dir=${ops_dir}/src

ops_bin_dir=`python -c "import easy_rec; print(easy_rec.get_ops_dir())" |tail -1`

if [ -z "$ops_bin_dir" ]
then
   echo "could not determine ops_bin_dir"
   exit 1
fi

if [ ! -e $ops_bin_dir ]
then
   mkdir -p $ops_bin_dir
fi

ops_bin=${ops_bin_dir}/libload_embed.so

g++ -D_GLIBCXX_USE_CXX11_ABI=$TF_ABI -shared -O3 -DNDEBUG -Wl,-rpath,'$ORIGIN'  -fpermissive -mfma -fopenmp  ${ops_src_dir}/load_kv_embed.cc ${ops_src_dir}/load_dense_embed.cc -o ${ops_bin}  -fPIC -I $TF_INC $TF_LFLAGS -L/lib64

python -c "import tensorflow as tf; tf.load_op_library('$ops_bin')"
err_code=$?
if [ $err_code -ne 0 ]
then
   echo "build failed"
   exit $err_code
fi
