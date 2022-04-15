# Performance of demo model using Dense Embedding Layer #
The performance of demo model introduced in [Examples/DenseDemo](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/examples/dense_demo.html).

## Profiling commands ##
Add `--trace-fork-before-exec=true` if MPI or multiple CPU processes is used to collect the timelines for all GPUs.
```shell
nsys profile --trace=nvtx,cuda --sample=none --backtrace=none --cudabacktrace=none --cpuctxsw=none -f true -o profiling_filename \
python3 script.py --arguments
```

## Infrastructure ##
```text
TensorFlow 2.5
embedding_vec_size: 4
slot_num: 100
nnz_per_slot: 10
batchsize for single GPU: 8192
batchsize for 8 GPUs: 65536
DGX A100
NsightSystems-linux-cli-public-2021.2.1.58
```

## Performance Numbers ##
### end2end elapsed time (miliseconds) ###
|      | 1 GPU | 8 GPUs |
| ---- | ----  | ------ |
| Original TF | 179.85 | —— |
| SOK | 25.90 | 45.36 |

### Query per seconds ###
|      | 1 GPU | 8 GPUs |
| ---- | ----  | ------ |
| Original TF | 45548 | —— |
| SOK | 316269 | 1444925 |
