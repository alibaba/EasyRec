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

REGISTER_OP("GetNcclUniqueId")
    .Output("nccl_unique_id: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle output_shape = ctx->Vector(32);
      ctx->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("GenRandomSeed")
    .Attr("seed: int = 0")
    .Output("global_seed: int64")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle output_shape = ctx->Vector(1);
      ctx->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("PluginInit")
    .Input("global_replica_id: int32")
    .Input("num_replicas_in_sync: int32")
    .Input("nccl_unique_id: int32")
    .Input("global_seed: int64")
    .Input("visible_devices: int32")
    .Attr("global_batch_size: int >= 1")
    .Output("status: string")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle input_shape_0 = ctx->input(0);
      DimensionHandle input_num_elem_0 = ctx->NumElements(input_shape_0);
      if (1 != ctx->Value(input_num_elem_0))
        return errors::InvalidArgument("global_replica_id must be a scalar.");

      ShapeHandle input_shape_1 = ctx->input(1);
      DimensionHandle input_num_elem_1 = ctx->NumElements(input_shape_1);
      if (1 != ctx->Value(input_num_elem_1))
        return errors::InvalidArgument("num_replicas_in_sync must be a scalar.");

      ShapeHandle input_shape_2 = ctx->input(2);
      int rank = ctx->Rank(input_shape_2);
      long long value_num = ctx->Value(ctx->NumElements(input_shape_2));
      if (1 != rank || 32 != value_num)
        return errors::InvalidArgument(
            "nccl_unique_id must be a vector with 32 elements, whose rank is 1.");

      ShapeHandle input_shape_3 = ctx->input(3);
      DimensionHandle input_num_elem_3 = ctx->NumElements(input_shape_3);
      if (1 != ctx->Value(input_num_elem_3))
        return errors::InvalidArgument("global_seed must be a scalar.");

      ShapeHandle visible_devices_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(4), 1, &visible_devices_shape));

      return Status::OK();
    });
