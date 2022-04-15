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
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

Status ReadEmbeddingVariableShapeFn(InferenceContext* c) {
  std::vector<ShapeAndType> shape_and_type;
  TF_RETURN_IF_ERROR(shape_inference::ValidateVariableResourceHandle(c, &shape_and_type));
  c->set_output(0, shape_and_type[0].shape);
  if (shape_and_type[0].dtype == DT_VARIANT && shape_and_type.size() > 1) {
    std::vector<ShapeAndType> variant_shape_and_type;
    std::copy(shape_and_type.begin() + 1, shape_and_type.end(),
              std::back_inserter(variant_shape_and_type));
    c->set_output_handle_shapes_and_types(0, variant_shape_and_type);
  }
  return Status::OK();
}

REGISTER_OP("ReadEmbeddingVariableOp")
    .Input("resource: resource")
    .Input("tf_resource: resource")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("unique_var_name: string")
    .SetShapeFn(ReadEmbeddingVariableShapeFn);