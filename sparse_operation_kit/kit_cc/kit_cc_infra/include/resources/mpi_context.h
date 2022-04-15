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

#ifndef RESOURCES_MPI_CONTEXT_H
#define RESOURCES_MPI_CONTEXT_H

#include <mpi.h>

#include <cstdint>
#include <memory>

namespace SparseOperationKit {

/*This class is used to manage MPI related things for SOK*/
class MPIContext {
 public:
  static std::unique_ptr<MPIContext> create(const uint32_t ranks);
  ~MPIContext();

  MPIContext() = delete;
  MPIContext(const MPIContext&) = delete;
  MPIContext& operator=(const MPIContext&) = delete;
  MPIContext(MPIContext&&) = delete;
  MPIContext& operator=(MPIContext&&) = delete;

  bool sync_via_mpi() const;
  int32_t rank_id() const;
  int32_t rank_size() const;
  void barrier() const;

 private:
  explicit MPIContext(const uint32_t ranks);

  const uint32_t ranks_;
  int32_t rank_id_;
  MPI_Comm my_comm_;
  const bool sync_via_mpi_;
  // whether MPI runtime is initialized by SOK.
  bool should_free_mpi_rt_;
};

using MPIContext_t = std::unique_ptr<MPIContext>;

}  // namespace SparseOperationKit

#endif  // RESOURCES_MPI_CONTEXT_H