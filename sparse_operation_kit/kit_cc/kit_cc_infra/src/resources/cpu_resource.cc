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

#include "resources/cpu_resource.h"

#include <cstdlib>

namespace SparseOperationKit {

int32_t GetWorkerThreadsCount() {
  auto atoi = [](const char* str, int32_t* num) -> bool {
    *num = std::atoi(str);
    if (*num < 1) return false;
    return true;
  };
  const auto sok_worker_threads_cnt = std::getenv("SOK_WORKER_THREADS_CNT");
  int32_t num = 1;
  return (sok_worker_threads_cnt && atoi(sok_worker_threads_cnt, &num)) ? num : 1;
}

CpuResource::Barrier::Barrier(const size_t thread_count)
    : mu_(), cond_(), thread_count_(thread_count), count_(thread_count), generation_(0) {}

void CpuResource::Barrier::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  auto local_gen = generation_;
  if (!--count_) {
    generation_++;
    count_ = thread_count_;
    cond_.notify_all();
  } else {
    cond_.wait_for(lock, time_threshold_, [this, local_gen]() { return local_gen != generation_; });
    if (local_gen == generation_)
      throw std::runtime_error(ErrorBase + "Intra-process barrier blocking threads time out.");
  }
}

CpuResource::BlockingCallOnce::BlockingCallOnce(const size_t thread_count)
    : mu_(), cond_(), thread_count_(thread_count), count_(thread_count), generation_(0) {}

CpuResource::CpuResource(const size_t local_gpu_cnt, const size_t global_gpu_cnt)
    : local_gpu_count_(local_gpu_cnt),
      global_gpu_count_(global_gpu_cnt),
      barrier_(std::make_shared<Barrier>(local_gpu_cnt)),
      blocking_call_oncer_(std::make_shared<BlockingCallOnce>(local_gpu_cnt)),
      mu_(),
      thread_pool_(new Eigen::SimpleThreadPool(local_gpu_cnt)),
      workers_(local_gpu_cnt),
      mpi_context_(nullptr) {
#ifdef SOK_ASYNC
  for (size_t dev_id = 0; dev_id < local_gpu_count_; dev_id++) {
    workers_[dev_id].reset(new Eigen::SimpleThreadPool(GetWorkerThreadsCount()));
  }
#endif
  const size_t node_num = global_gpu_cnt / local_gpu_cnt;
  mpi_context_.reset(MPIContext::create(/*ranks=*/node_num).release());
}

std::shared_ptr<CpuResource> CpuResource::Create(const size_t local_gpu_cnt,
                                                 const size_t global_gpu_cnt) {
  return std::shared_ptr<CpuResource>(new CpuResource(local_gpu_cnt, global_gpu_cnt));
}

void CpuResource::sync_cpu_threads() const { barrier_->wait(); }

void CpuResource::sync_threadpool() const {
  while (!thread_pool_->Done()) std::this_thread::yield();
}

void CpuResource::sync_all_workers_via_mpi() const { mpi_context_->barrier(); }

}  // namespace SparseOperationKit