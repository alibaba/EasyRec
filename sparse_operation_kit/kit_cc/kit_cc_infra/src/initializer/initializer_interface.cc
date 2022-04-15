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

#include "initializer/initializer_interface.h"

#include <unordered_map>

#include "common.h"
#include "initializer/constant_initializer.h"
#include "initializer/random_uniform.h"
#include "resources/manager.h"

namespace SparseOperationKit {

const std::unordered_map<std::string, InitializerType> InitializerMap = {
    {"random_uniform", InitializerType::RandomUniform},
    {"ones", InitializerType::Ones},
    {"zeros", InitializerType::Zeros}};

std::shared_ptr<Initializer> Initializer::Get(const std::string initializer) {
  // Get initializer enum type
  InitializerType initializer_type = InitializerType::Ones;
  find_item_in_map(InitializerMap, initializer, initializer_type);

  // generate corresponding initializer instance.
  switch (initializer_type) {
    case InitializerType::RandomUniform: {
      return RandomUniformInit::create(/*a=*/-0.05f, /*b=*/0.05f);
    }
    case InitializerType::Ones: {
      return ConstantInit::create(1.0f);
    }
    case InitializerType::Zeros: {
      return ConstantInit::create(0.0f);
    }
    default: {
      break;
    }
  }  // switch initializer_type

  throw std::runtime_error(ErrorBase + "Not supported initializer.");
}

}  // namespace SparseOperationKit