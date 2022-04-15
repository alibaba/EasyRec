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

REGISTER_OP("CreateEmbeddingDense")
    .Input("emb_var_handle: resource")
    .Attr("input_dispatcher: string")
    .Attr("input_dispatcher_subsequent_ops: list(string) = []")
    .Attr("embedding_lookuper: string")
    .Attr("output_dispatcher: string")
    .Attr("output_dispatcher_subsequent_ops: list(string) = []")
    .Output("emb_layer_handle: variant")
    .Attr("slot_num: int >= 1 = 1")
    .Attr("nnz_per_slot: int >= 1 = 1")
    .Attr("layer_handle_name: string")
    .Attr("compute_dtype: {float32, float16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle output_shape = ctx->Scalar();
      ctx->set_output(0, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
        This operation is used to create embedding layer that will not 
        do reduction intra slots, which means embedding vectors will
        be concatenated.
    )doc");
