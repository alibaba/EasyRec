set -e 
export PS4='\n\033[0;33m+[${BASH_SOURCE}:${LINENO}]: \033[0m'
set -x

## ============================= single GPU =============================== ##

# dense embedding + adam + save_param + hashtable
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --mixed_precision=1

# dense embedding + compat adam + save_param + hashtable
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

# dense embedding + plugin_adam + restore_param + hashtable
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=0 \
    --restore_params=1 \
    --use_hashtable=1 \
    --key_dtype="uint32"
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=0 \
    --restore_params=1 \
    --use_hashtable=1 \
    --mixed_precision=1

# dense embedding + adam + no-hashtable
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=0 \
    --restore_params=0 \
    --use_hashtable=0
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=0 \
    --restore_params=0 \
    --use_hashtable=0 \
    --mixed_precision=1

# dense embedding + adam + dynamic_input
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=0 \
    --restore_params=0 \
    --use_hashtable=0 \
    --dynamic_input=1

# dense embedding + adam + multi embedding layers
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num 20 10 \
    --nnz_per_slot=10 \
    --embedding_vec_size 4 8 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num 20 10 \
    --nnz_per_slot=10 \
    --embedding_vec_size 4 8 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --mixed_precision=1

# sparse embedding + compat_adam + save_params + hashtable + combiner=mean
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --combiner="mean" \
    --use_hashtable=1
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --combiner="mean" \
    --use_hashtable=1 \
    --mixed_precision=1

# sparse embedding + plugin_adam + save_params + hashtable + combiner=mean
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --combiner="mean" \
    --use_hashtable=1
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --combiner="mean" \
    --use_hashtable=1 \
    --mixed_precision=1

# sparse embedding + plugin_adam + save_params + hashtable + combiner=sum
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

# sparse embedding + adam + restore_params + hashtable
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=0 \
    --restore_params=1 \
    --use_hashtable=1 \
    --key_dtype="uint32"

# sparse embedding + adam + save_params + no-hashtable
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=0

# sparse embedding + plugin_adam + multi-embedding layers
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num 10 20 \
    --max_nnz=10 \
    --embedding_vec_size 4 8 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num 10 20 \
    --max_nnz=10 \
    --embedding_vec_size 4 8 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --mixed_precision=1

## ============================================= horovod ======================== #

# dense embedding + compat_adam + save_params + hashtable
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_dense_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --functional_api=1

mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_dense_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --key_dtype='uint32' \
    --use_tf_initializer=1
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_dense_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --mixed_precision=1

# dense embedding + adam + save_params + hashtable
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_dense_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

# dense embedding + adam + save_params + multi-embedding layers
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_dense_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num 20 10 \
    --nnz_per_slot=10 \
    --embedding_vec_size 4 8 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

# sparse_embedding + compat_adam + save_params + hashtable
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_sparse_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --functional_api=1

mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_sparse_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --use_tf_initializer=1
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_sparse_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="compat_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1 \
    --mixed_precision=1

# sparse_embedding + adam + save_params + hashtable
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_sparse_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

# sparse embedding + plugin_adam + save_params + multi-embedding layers
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_sparse_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num 10 20 \
    --max_nnz=10 \
    --embedding_vec_size 4 8 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1


# ====== clean intermediate files ========== #
rm -rf *.file && rm -rf embedding_variables/