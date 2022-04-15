# Known Issues #
There are several issues in SparseOperationKit, and we are trying to fix shose issues in the near future.

## NCCL conflicts ##
In SparseOperationKit's embedding layers, NCCL is used to transfer data among GPUs. When there exists multiple embedding layers and there is not data dependencies among those layers, the execution order must be deterministic otherwise program might be hanging.
```text
device-0: embedding-0 -> embedding-1
device-1: embedding-1 -> embedding-0
``` 
The solution for such problem is to make the program launch those layers with the same order in different GPUs, you can add `tf.control_dependencies()` between different SOK embedding layers to force the deterministic launching order.