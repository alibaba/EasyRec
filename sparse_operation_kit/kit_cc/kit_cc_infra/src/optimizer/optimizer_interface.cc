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

#include "optimizer/optimizer_interface.h"

#include "optimizer/adam_optimizer.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace SparseOperationKit {

std::shared_ptr<Optimizer> Optimizer::Get(const OptimizerType optimizer_type,
                                          optimizer_hyper_params&& hyper_params,
                                          const std::shared_ptr<ParamsManager>& params_mgr,
                                          std::shared_ptr<ResourcesManager>& resource_mgr) {
  switch (optimizer_type) {
    case OptimizerType::Adam: {
      return AdamOptimizer::create(std::move(hyper_params), params_mgr, resource_mgr);
    }
    default: {
      break;
    }
  }  // switch optimizer_type
  throw std::runtime_error(ErrorBase + "Not supported optimizer type.");
}

Optimizer::Optimizer(const std::shared_ptr<ParamsManager>& params_mgr) : params_mgr_(params_mgr) {}

class OptimizerVariantWrapper {
 public:
  OptimizerVariantWrapper() : optimizer_(nullptr) {}
  explicit OptimizerVariantWrapper(const std::shared_ptr<Optimizer> optimizer)
      : optimizer_(optimizer) {}
  OptimizerVariantWrapper(const OptimizerVariantWrapper& other) : optimizer_(other.optimizer_) {}
  OptimizerVariantWrapper& operator=(OptimizerVariantWrapper&& other) {
    if (&other == this) return *this;
    optimizer_ = other.optimizer_;
    return *this;
  }
  OptimizerVariantWrapper& operator=(const OptimizerVariantWrapper& other) = delete;

  std::shared_ptr<Optimizer> get() const { return optimizer_; }

  ~OptimizerVariantWrapper() = default;
  tensorflow::string TypeName() const { return "EmbeddingPlugin::OptimizerVariantWrapper"; }
  void Encode(tensorflow::VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "OptimizerVariantWrapper objects.";
  }
  bool Decode(const tensorflow::VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "OptimizerVariantWrapper objects.";
    return false;
  }

 private:
  std::shared_ptr<Optimizer> optimizer_;
};

void StoreOptimizerInVariantTensor(const std::shared_ptr<Optimizer>& optimizer,
                                   tensorflow::Tensor* tensor) {
  if (!(tensor->dtype() == tensorflow::DT_VARIANT &&
        tensorflow::TensorShapeUtils::IsScalar(tensor->shape())))
    throw std::runtime_error(ErrorBase + "Optimizer tensor must be a scalar of dtype DT_VARIANT.");

  tensor->scalar<tensorflow::Variant>()() = OptimizerVariantWrapper(optimizer);
}

}  // namespace SparseOperationKit