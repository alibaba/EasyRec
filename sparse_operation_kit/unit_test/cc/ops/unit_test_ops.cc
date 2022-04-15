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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("AllGatherDispatcher")
    .Input("global_replica_id: int32")
    .Input("num_replicas_in_sync: int32")
    .Input("values: value_dtype")
    .Input("indices: indice_dtype")
    .Output("values_out: value_dtype")
    .Output("indices_out: indice_dtype")
    .Output("num_elements: int64")
    .Output("total_valid_num: int64")
    .Attr("value_dtype: {int64}")
    .Attr("indice_dtype: {int64}")
    .Attr("global_batch_size: int")
    .Attr("rows_num_per_sample: int")
    .Attr("max_nnz: int");

REGISTER_OP("CsrConversionDistributed")
    .Input("global_replica_id: int32")
    .Input("values: int64")
    .Input("row_indices: int64")
    .Input("total_valid_num: int64")
    .Output("replica_values: int64")
    .Output("replica_csr_row_offsets: int64")
    .Output("replica_nnz: int64")
    .Attr("global_batch_size: int")
    .Attr("slot_num: int")
    .Attr("max_nnz: int");

REGISTER_OP("ReduceScatterDispatcher")
    .Input("global_replica_id: int32")
    .Input("input: float")
    .Output("output: float")
    .Attr("global_batch_size: int")
    .Attr("slot_num: int")
    .Attr("max_nnz: int");