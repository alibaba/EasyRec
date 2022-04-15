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

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("PluginBprop")
    .Input("emb_handle: variant")
    .Input("global_replica_id: int32")
    .Input("top_gradient: dtype")
    .Input("replica_nnz: uint64")
    .Output("gradient: dtype")
    .Output("value_index: int64")
    .Attr("dtype: {float32, float16}")
    .Attr("unique_op_name: string")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
        the output value_index is the hash_value_index_tensor used internally.

        This op is used to calculate the gradient for embedding variable based
        on the top_gradients.

        'replica_nnz' represents how many keys are processed in this step by this GPU.
    )doc");