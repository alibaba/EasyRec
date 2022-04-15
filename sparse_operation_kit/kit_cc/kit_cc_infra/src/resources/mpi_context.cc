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

#include "resources/mpi_context.h"

#include "common.h"

namespace SparseOperationKit {

namespace {

bool GetSyncViaMpi() {
  const auto sok_sync_via_mpi = std::getenv("SOK_SYNC_VIA_MPI");
  if (nullptr != sok_sync_via_mpi && 1 == std::atoi(sok_sync_via_mpi)) {
    return true;
  } else {
    return false;
  }
}

// if not, throw error
#define ENSURE_MPI_USED(ctx)                                 \
  do {                                                       \
    if (!(ctx)->sync_via_mpi()) {                            \
      throw std::runtime_error(ErrorBase + "MPI not used."); \
    }                                                        \
  } while (0)

// if not, just return
#define SOFT_ENSURE_MPI_USED(ctx) \
  do {                            \
    if (!(ctx)->sync_via_mpi()) { \
      return;                     \
    }                             \
  } while (0)

}  // anonymous namespace

MPIContext::MPIContext(const uint32_t ranks)
    : ranks_(ranks), rank_id_(-1), sync_via_mpi_{GetSyncViaMpi()}, should_free_mpi_rt_{false} {
  if (sync_via_mpi_) {
    int32_t init_flag = 0;
    CK_MPI(MPI_Initialized(&init_flag));
    if (0 == init_flag) {
      // MPI runtime should be released by me.
      should_free_mpi_rt_ = true;

      int32_t provided = MPI_THREAD_SINGLE;
      CK_MPI(MPI_Init_thread(nullptr, nullptr, /*required=*/MPI_THREAD_MULTIPLE, &provided));
      my_comm_ = MPI_COMM_WORLD;
    } else {
      // MPI runtime should not be released by me.
      should_free_mpi_rt_ = false;

      // duplicate the world_comm
      CK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &my_comm_));
    }
    int32_t rank_size = 0;
    CK_MPI(MPI_Comm_size(my_comm_, &rank_size));
    if (ranks_ != rank_size)
      throw std::runtime_error(ErrorBase + "ranks is not equal to that of MPI_Comm_size.");
    CK_MPI(MPI_Comm_rank(my_comm_, &rank_id_));
    CK_MPI(MPI_Comm_set_errhandler(my_comm_, MPI_ERRORS_RETURN));
    int32_t provided = MPI_THREAD_SINGLE;
    CK_MPI(MPI_Query_thread(&provided));
    if (provided < MPI_THREAD_MULTIPLE)
      throw std::runtime_error(ErrorBase +
                               "MPI been initialized without multi-threading support"
                               " [MPI_THREAD_MULTIPLE], which will likely leads to seg fault.");
  }
}

MPIContext::~MPIContext() {
  try {
    if (sync_via_mpi_ && !should_free_mpi_rt_) {
      CK_MPI(MPI_Comm_free(&my_comm_));
    }
    int32_t free_flag = 0;
    CK_MPI(MPI_Finalized(&free_flag));
    if (0 == free_flag && should_free_mpi_rt_) {
      CK_MPI(MPI_Finalize());
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
  }
}

std::unique_ptr<MPIContext> MPIContext::create(const uint32_t ranks) {
  return std::unique_ptr<MPIContext>(new MPIContext(ranks));
}

bool MPIContext::sync_via_mpi() const { return sync_via_mpi_; }

int32_t MPIContext::rank_id() const {
  ENSURE_MPI_USED(this);
  return rank_id_;
}

int32_t MPIContext::rank_size() const {
  ENSURE_MPI_USED(this);
  return ranks_;
}

void MPIContext::barrier() const {
  SOFT_ENSURE_MPI_USED(this);
  CK_MPI(MPI_Barrier(my_comm_));
}

}  // namespace SparseOperationKit