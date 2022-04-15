# DLRM using SparseOperationKit #
Demonstrates how to build DLRM model with SparseOperationKit.

You can find the source codes in [`sparse_operation_kit/documents/tutorials/DLRM/`](https://github.com/NVIDIA/HugeCTR/tree/master/sparse_operation_kit/documents/tutorials/DLRM).

## steps ##
### Generate datasets ###
[Criteo Terabytes Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) will be used. Download these files. And there are several options for you to generate datasets.
#### [Option1] ####
Follow [TensorFlow's instructions](https://github.com/tensorflow/models/tree/master/official/recommendation/ranking/preprocessing) to process these files and save as CSV files.

#### [Option2] ####
Follow [HugeCTR's instructions](https://github.com/NVIDIA/HugeCTR/tree/master/samples/dlrm#preprocess-the-terabyte-click-logs) to process these files. Then convert the generated binary files to CSV files.
```shell
$ python3 bin2csv.py \
    --input_file="YourBinaryFilePath/train.bin" \
    --num_output_files=1024 \
    --output_path="./train/" \
    --save_prefix="train_"
```
```shell
$ python3 bin2csv.py \
    --input_file="YourBinaryFilePath/test.bin" \
    --num_output_files=64 \
    --output_path="./test/" \
    --save_prefix="test_"
```

### Set common params ###
```shell
$ export EMBEDDING_DIM=32
```

### Run DLRM with TensorFlow ###
```shell
$ mpiexec --allow-run-as-root -np 4 \
    python3 main.py \
        --global_batch_size=16384 \
        --train_file_pattern="./train/*.csv" \
        --test_file_pattern="./test/*.csv" \
        --embedding_layer="TF" \
        --embedding_vec_size=$EMBEDDING_DIM \
        --bottom_stack 512 256 $EMBEDDING_DIM \
        --top_stack 1024 1024 512 256 1 \
        --distribute_strategy="multiworker" \
        --TF_MP=1
```

### Run DLRM with SOK ###
```shell
$ mpiexec --allow-run-as-root -np 4 \
    python3 main.py \
        --global_batch_size=16384 \
        --train_file_pattern="./train/*.csv" \
        --test_file_pattern="./test/*.csv" \
        --embedding_layer="SOK" \
        --embedding_vec_size=$EMBEDDING_DIM \
        --bottom_stack 512 256 $EMBEDDING_DIM \
        --top_stack 1024 1024 512 256 1 \
        --distribute_strategy="multiworker"
```

## reference ##
1. DLRM (https://arxiv.org/pdf/1906.00091.pdf)
2. Criteo TeraBytes Datasets (https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
3. TensorFlow DLRM model (https://github.com/tensorflow/models/tree/master/official/recommendation/ranking)