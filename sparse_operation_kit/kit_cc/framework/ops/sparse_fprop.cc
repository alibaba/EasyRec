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

namespace OP_OVERLOAD { 
namespace {
Status ValidateVariableResourceHandle(
    InferenceContext* c, std::vector<ShapeAndType>* shape_and_type) {
  {
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr("dtype", &value_dtype));
    if (DataType::DT_FLOAT == value_dtype) 
      return shape_inference::ValidateVariableResourceHandle(c, shape_and_type);
  }

  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    shape_and_type->emplace_back(c->UnknownShape(), DT_INVALID);
  } else {
    *shape_and_type = *handle_data;
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr("dtype", &value_dtype));
    if (shape_and_type->at(0).dtype != value_dtype) {
      if (DataType::DT_HALF == value_dtype) {
        return Status::OK();
      } else {
        return errors::InvalidArgument(
          "Trying to read variable with wrong dtype. "
          "Expected ",
          DataTypeString(shape_and_type->at(0).dtype), " got ",
          DataTypeString(value_dtype));
      } // if DT_HALF == value_dtype
    } // if DataType != value_dtype
  }
  return Status::OK();
}
} // anonymous namespace
} // namespace OP_OVERLOAD

REGISTER_OP("PluginSparseFprop")
    .Input("emb_var_handle: resource")
    .Input("emb_handle: variant")
    .Input("values: value_dtype")
    .Input("indices: indice_dtype")
    .Input("global_replica_id: int32")
    .Output("emb_vector: dtype")
    .Output("replica_nnz: uint64")
    .Attr("slot_num: int")
    .Attr("training: bool")
    .Attr("value_dtype: {uint32, int64}")
    .Attr("indice_dtype: {int64}")
    .Attr("dtype: {float32, float16}")
    .Attr("unique_op_name: string")
    .SetShapeFn([](InferenceContext* ctx) {
      std::vector<ShapeAndType> handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          OP_OVERLOAD::ValidateVariableResourceHandle(ctx, &handle_shape_and_type));

      ShapeHandle variable_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(handle_shape_and_type[0].shape, 2, &variable_shape));
      DimensionHandle emb_vec_size_dim = ctx->Dim(variable_shape, 1);

      tensorflow::int64 slot_num = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("slot_num", &slot_num));
      DimensionHandle slot_num_dim = ctx->MakeDim(slot_num);

      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, slot_num_dim, emb_vec_size_dim});
      ctx->set_output(0, output_shape);

      return Status::OK();
    })
    .Doc(R"doc(
        This op can be used for all kinds of embedding forward propagation,
        which requires the unique_op_name to identify different op instance.
        For example:
            vec0 = plugin_fprop(emb_handle0, values, indices, unique_op_name='1')
            vec1 = plugin_fprop(emb_handle1, values, indices, unique_op_name='2')

            where different unique_op_name are set for different embedding layer instance.
        
         'replica_nnz' represents how many keys are processed in this step by this GPU.
    )doc");