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

REGISTER_OP("AssignEmbeddingVariable")
    .Input("emb_var_handle: resource")
    .Input("tf_var_handle: resource")
    .Input("initial_value: init_dtype")
    .Input("local_replica_id: int32")
    .Attr("var_name: string")
    .Attr("trainable: bool = true")
    .Attr("shape: shape")
    .Attr("use_hashtable: bool = true")
    .Attr("dtype: type")
    .Attr("init_dtype: {float, string}")
    .Attr("key_dtype: {uint32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      std::vector<ShapeAndType> handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          shape_inference::ValidateVariableResourceHandle(c, &handle_shape_and_type));

      return Status::OK();
    })
    .Doc(R"doc(
        This op is used as the initializer op of embedding variable.
    )doc");
