/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPERATION_INTERFACE_H
#define OPERATION_INTERFACE_H

/*
 * Sparse Embedding Algorithm:
 *   - Input tensors:
 *       Forward:
 *           1. replica_values, which is the keys for current GPU.
 *           2. replica_indices, which is the coo-row-indices of the replica_values.
 *       Backward:
 *           1. replica_top_gradient, which is the top gradients tensor for embedding
 *               vector of current GPU.
 *   - Output tensors:
 *       Forward:
 *           1. replica_output, which is the output tensor of this embedding layer
 *               in current GPU. its shape is [replica_batch_size, slot_num, embedding_vec_size].
 *       Backward:
 *           1. value_index_tensor, which is the local indices of input gradients.
 *               whose type is int64, and shape is [replica_host_nnz,]
 *           2. replica_input_grad, which is the input gradients for current GPU.
 *               whose shape is [replica_host_nnz, embedding_vec_size]
 *   - Notes:
 *       1. embedding lookuper must produce an 1-dim host tensor for each GPU, whose name is
 *           replica_host_nnz and type is size_t. This tensor will be used to generate the
 *           gradients shape for current GPU. If embedding lookuper does not produce such
 *           tensor, then one of the other operations must produce this tensor.
 *
 * Dense Embedding Algorithm:
 *   - Input tensors:
 *       Forward:
 *           1. replica_values, which is the keys for current GPU.
 *       Backward:
 *           1. replica_top_gradient, save to Sparse Embedding Algorithm.
 *   - Output tensors:
 *       Forward:
 *           1. replica_output, which is the output tensor of this embedding layer in
 *               current GPU. Its shape is:
 *                   [replica_batch_size, slot_num, nnz_per_slot, embedding_vec_size].
 *       Backward:
 *           1. value_index_tensor, same to Sparse Embedding Algorithm.
 *           2. replica_input_grad, same to Sparse Embedding Algorithm.
 *
 *   - Notes:
 *       1. embedding lookuper must produce an 1-dim host tensor for each GPU, whose name is
 *           replica_host_nnz and type is size_t. This tensor will be used to generate the
 *           gradients shape for current GPU. If embedding lookuper does not produce such
 *           tensor, then one of the other operations must produce this tensor.
 */

/*
 * This file is the interface header for expanding operations.
 *
 * In this framework, an embedding layer is made of there key components,
 * which are 'input_dispathcer', 'embedding_lookuper', 'output_dispatcher',
 * respectively.
 *
 * 1. Input_dispathcer will dispatch input data to each GPU,
 * it will convert the computation from data-parallel to model-parallel.
 * 2. Embedding_lookuper is responsible for local-gpu computation, and looking
 * up embedding vector, which is key->embedding_vector.
 * 3. Output_dispatcher will dispatch output data to each GPU,
 * it will convert the computation from model-parallel to data-parallel.
 *
 * To increase the flexibility of this framework, operations can be linked
 * to input_dispathcer or output_dispatcher. For example, the computation pipeline
 * might look like:
 *   1. input_dispathcer->embedding_lookuper->output_dispatcher
 *   2. input_dispathcer->op1->op2->op3->embedding_lookuper->output_dispatcher->op4->op5
 * The number of ops after input_dispathcer and output_dispatcher is not limited.
 *
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * The following steps describes how to add new operations to this framework.
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * ----------------------------------
 *   input / output dispathcer
 * ----------------------------------
 * 1. create a cpp source file. For example, MyInputDispathcer.cc
 * 2. "operation_interface.h" should be included in that souce file.
 * 3. inherit from Dispatcher class, and override methods:
 *   allocate_forward_spaces, allocate_backward_spaces, forward, backward
 *   for example:
 *   class MyInputDispathcer : public Dispatcher {
 *   public:
 *       explicit MyInputDispathcer(ConstructionContext_t context) : Dispatcher(context) {}
 *       void allocate_forward_spaces() override {
 *           // reserve spaces used in forward propagation.
 *           // this function will be called only once, so that forward spaces used by all local
 * GPUs
 *           // should be reserved.
 *           // This function will be called once in each CPU process.
 *       }
 *       void allocate_backward_spaces() override {
 *           // reserve spaces used in backward propagation.
 *           // this function will be called only once, so that backward spaces used by all local
 * GPUs
 *           // should be reserved.
 *           // This function will be called once in each CPU process.
 *       }
 *       void forward(const Context_t &replica_context, const bool training) override {
 *           // do forward propagation.
 *           // this function will be called by multiple CPU-threads if there are multiple GPUs.
 *           // This function will be called by each CPU threads.
 *       }
 *       void backward(const Context_t &replica_context) override {
 *           // do backward propagation.
 *           // this function will be called by multiple CPU-threads if there are multiple GPUs.
 *           // This function will be called by each CPU threads.
 *       }
 *       [Optional]:
 *       bool dump(DumpCallBack dump_call_back) const override {
 *           // if this dispatcher has something needed to be dump to file,
 *           // then override this virtual function, and return true!!
 *           //
 *           // if this virtual function is overrided, then write the logics of
 *           // dumping internal states to a dump_call_back function.
 *           // DumpCallBack is a function object which only takes std::ofstream& as
 *           // its input.
 *           // This function will be called by each CPU process.
 *           // usage example:
 *           bool dump(DumpCallBack dump_call_back) const override {
 *               dump_call_back = [](std::ofstream &file_stream) {
 *                   std::cout << "the dump call back function is called." << std::endl;
 *               };
 *               // important: this overrided function must return true, otherwise its
 *               // dump_call_back will not be called.
 *               return true;
 *           }
 *       }
 *       void restore(const std::ifstream &filestream) override {
 *           // if this operation has something needed to be restored from file,
 *           // then override this virtual function.
 *           // This function will be called by each CPU process.
 *       }
 *       void load_embedding_values(const std::vector<std::shared_ptr<Tensor>> &tensor_list)
 * override {
 *           // if this operation has something needed to be modified with embedding values,
 *           // then override this virtual function.
 *           // This function will be called by each CPU process.
 *       }
 *   };
 *
 * 4. register this dispatcher by calling 'REGISTER_INPUT_DISPATCHER_BUILDER' macro in the cpp file.
 *   for example: 
 *      REGISTER_INPUT_DISPATCHER_BUILDER("MyInputDispathcer", 
 *                                        DataType::Int64, DataType::Float32, 
 *                                        MyInputDispathcer);
 * 5. The name 'MyInputDispathcer', key_dtype 'DataType::Int64', and the dtype 'DataType::Float32' will be 
 *   used together as the identifier to identify this class.
 *
 * ----------------------------------
 *        embedding_lookuper
 * ----------------------------------
 * 1. create a cpp source file. For example, MyLookuper.cc
 * 2. "operation_interface.h" should be included in that souce file.
 * 3. inherit from EmbeddingLookuper class, and override methods:
 *   allocate_forward_spaces, allocate_backward_spaces, forward, backward,
 *   load_tensors
 *   for example:
 *   class MyLookuper : public EmbeddingLookuper {
 *   public:
 *       MyLookuper(ConstructionContext_t context, std::shared_ptr<ParamInterface> param)
 *            : EmbeddingLookuper(context, param) {}
 *       void allocate_forward_spaces() override {
 *           // reserve spaces used in forward propagation.
 *           // this function will be called only once, so that forward spaces used by all local
 * GPUs
 *           // should be reserved.
 *           // This function will be called by each CPU process.
 *       }
 *       void allocate_backward_spaces() override {
 *           // reserve spaces used in backward propagation.
 *           // this function will be called only once, so that backward spaces used by all local
 * GPUs
 *           // should be reserved.
 *           // This function will be called by each CPU process.
 *       }
 *       void forward(const Context_t &replica_context, const bool training) override {
 *           // do forward propagation.
 *           // this function will be called by multiple CPU-threads if there are multiple GPUs.
 *           // This function will be called by each CPU threads.
 *       }
 *       void backward(const Context_t &replica_context) override {
 *           // do backward propagation.
 *           // this function will be called by multiple CPU-threads if there are multiple GPUs.
 *           // This function will be called by each CPU threads.
 *       }
 *       void save_params(std::shared_ptr<Tensor> &keys,
 *                        std::shared_ptr<Tensor> &embedding_values,
 *                        size_t &num_total_keys) const override {
 *           // the embedding lookup has to override this function to help
 *           // param saving the parameters from GPU memory to files.
 *           // this function has to aggregate parameters from all GPUs to
 *           // the CPU tensors: keys & embedding_values
 *       }
 *       void restore_params(const std::shared_ptr<Tensor> &keys,
 *                           const std::shared_ptr<Tensor> &embedding_values,
 *                           const size_t num_total_keys) override {
 *           // the embedding lookuper has to override this function to help
 *           // param restoring the parameters from CPU keys and embedding values
 *           // to param's GPU memory.
 *           // This function will be called by each CPU process.
 *       }
 *       [Optional]:
 *       bool dump(DumpCallBack dump_call_back) const override {
 *           // if this dispatcher has something needed to be dump to file,
 *           // then override this virtual function, and return true!!
 *           //
 *           // if this virtual function is overrided, then write the logics of
 *           // dumping internal states to a dump_call_back function.
 *           // DumpCallBack is a function object which only takes std::ofstream& as
 *           // its input.
 *           // This function will be called by each CPU process.
 *           // usage example:
 *           bool dump(DumpCallBack dump_call_back) const override {
 *               dump_call_back = [](std::ofstream &file_stream) {
 *                   std::cout << "the dump call back function is called." << std::endl;
 *               };
 *               // important: this overrided function must return true, otherwise its
 *               // dump_call_back will not be called.
 *               return true;
 *           }
 *       }
 *       void restore(const std::ifstream &filestream) override {
 *           // if this operation has something needed to be restored from file,
 *           // then override this virtual function.
 *           // This function will be called by each CPU process.
 *       }
 *       void load_embedding_values(const std::vector<std::shared_ptr<Tensor>> &tensor_list)
 * override {
 *           // if this operation has something needed to be modified with embedding values,
 *           // then override this virtual function.
 *           // This function will be called by each CPU process.
 *       }
 *   };
 *
 * 4. register this embedding_lookuper by calling 'REGISTER_EMB_LOOKUPER_BUILDER' macro in the cpp
 * file. for example: 
 *      REGISTER_EMB_LOOKUPER_BUILDER("MyLookuper", 
 *                                    DataType::Int64, DataType::Float32, 
 *                                    MyLookuper);
 * 5. The name 'MyLookuper', key_dtype 'DataType::Int64', and the dtype 'DataType::Float32' will be 
 *   used together as the identifier to identify this class.
 *
 * ----------------------------------
 *           operation
 * ----------------------------------
 * 1. create a cpp source file. For example, MyOperation.cc
 * 2. "operation_interface.h" should be included in that souce file.
 * 3. inherit from Operation class, and override methods:
 *   allocate_forward_spaces, allocate_backward_spaces, forward, backward
 *   for example:
 *   class MyOperation : public Operation {
 *   public:
 *       explicit MyOperation(ConstructionContext_t context) : Operation(context) {}
 *       void allocate_forward_spaces() override {
 *           // reserve spaces used in forward propagation.
 *           // this function will be called only once, so that forward spaces used by all local
 * GPUs
 *           // should be reserved.
 *           // This function will be called by each CPU process.
 *       }
 *       void allocate_backward_spaces() override {
 *           // reserve spaces used in backward propagation.
 *           // this function will be called only once, so that backward spaces used by all local
 * GPUs
 *           // should be reserved.
 *           // This function will be called by each CPU process.
 *       }
 *       void forward(const Context_t &replica_context, const bool training) override {
 *           // do forward propagation.
 *           // this function will be called by multiple CPU-threads if there are multiple GPUs.
 *           // This function will be called by each CPU threads.
 *       }
 *       void backward(const Context_t &replica_context) override {
 *           // do backward propagation.
 *           // this function will be called by multiple CPU-threads if there are multiple GPUs.
 *           // This function will be called by each CPU threads.
 *       }
 *       [Optional]:
 *       bool dump(DumpCallBack dump_call_back) const override {
 *           // if this dispatcher has something needed to be dump to file,
 *           // then override this virtual function, and return true!!
 *           //
 *           // if this virtual function is overrided, then write the logics of
 *           // dumping internal states to a dump_call_back function.
 *           // DumpCallBack is a function object which only takes std::ofstream& as
 *           // its input.
 *           // This function will be called by each CPU process.
 *           // usage example:
 *           bool dump(DumpCallBack dump_call_back) const override {
 *               dump_call_back = [](std::ofstream &file_stream) {
 *                   std::cout << "the dump call back function is called." << std::endl;
 *               };
 *               // important: this overrided function must return true, otherwise its
 *               // dump_call_back will not be called.
 *               return true;
 *           }
 *       }
 *       void restore(const std::ifstream &filestream) override {
 *           // if this operation has something needed to be restored from file,
 *           // then override this virtual function.
 *           // This function will be called by each CPU process.
 *       }
 *       void load_embedding_values(const std::vector<std::shared_ptr<Tensor>> &tensor_list)
 * override {
 *           // if this operation has something needed to be modified with embedding values,
 *           // then override this virtual function.
 *           // This function will be called by each CPU process.
 *       }
 *   };
 *
 * 4. register this dispatcher by calling 'REGISTER_OPERATION_BUILDER' macro in the cpp file.
 *   for example: 
 *      REGISTER_OPERATION_BUILDER("MyOperation", 
 *                                 DataType::Int64, DataType::Float32, 
 *                                 MyOperation);
 * 5. The name 'MyOperation', key_dtype 'DataType::Int64', and the dtype 'DataType::Float32' will be 
 *   used together as the identifier to identify this class.
 */

#include "operation/operation.h"
#include "operation/operation_helper.h"
#include "tensor_buffer/tensor2_wrapper.h"

#endif  // OPERATION_INTERFACE_H