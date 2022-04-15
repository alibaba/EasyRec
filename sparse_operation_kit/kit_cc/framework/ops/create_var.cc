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

REGISTER_OP("CreateVar")
    .Attr("var_name: string")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Output("emb_var_handle: resource")
    .Output("tf_var_handle: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());

      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(0, std::vector<ShapeAndType>{{s, t}});
      c->set_output_handle_shapes_and_types(1, std::vector<ShapeAndType>{{s, t}});

      return Status::OK();
    })
    .Doc(R"doc(
        This op is used create variables used by a embedding layer on all GPUs in single worker.
        shape specify the variable's shape created on ONE GPU.
    )doc");