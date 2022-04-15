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

#ifndef CPU_RESOURCE_H
#define CPU_RESOURCE_H

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include "common.h"
#include "eigen3/unsupported/Eigen/CXX11/src/ThreadPool/SimpleThreadPool.h"
#include "resources/mpi_context.h"

namespace SparseOperationKit {

class CpuResource {
  class Barrier {
   public:
    explicit Barrier(size_t thread_count);

    Barrier() = delete;
    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;
    Barrier(Barrier&&) = delete;
    Barrier& operator=(Barrier&&) = delete;

    void wait();

   private:
    std::mutex mu_;
    std::condition_variable cond_;
    const size_t thread_count_;
    volatile size_t count_;
    volatile size_t generation_;
    const std::chrono::seconds time_threshold_{10};
  };

  class BlockingCallOnce {
   public:
    explicit BlockingCallOnce(const size_t thread_count);

    BlockingCallOnce() = delete;
    BlockingCallOnce(const BlockingCallOnce&) = delete;
    BlockingCallOnce& operator=(const BlockingCallOnce&) = delete;
    BlockingCallOnce(BlockingCallOnce&&) = delete;
    BlockingCallOnce& operator=(BlockingCallOnce&&) = delete;

    template <typename Callable, typename... Args>
    void operator()(Callable&& func, Args&&... args) {
      std::unique_lock<std::mutex> lock(mu_);
      auto local_gen = generation_;
      if (!--count_) {
        generation_++;
        count_ = thread_count_;

        /*call once in this generation*/
        auto bound_functor = std::bind(func, args...);
        once_callable_ = &bound_functor;
        once_call_ = &BlockingCallOnce::once_call_impl_<decltype(bound_functor)>;

        try {
          (this->*once_call_)();
          cond_.notify_all();
        } catch (...) {
          excp_ptr_ = std::current_exception();
          cond_.notify_all();  // TODO: Need anthoer mutex??
          std::rethrow_exception(excp_ptr_);
        }
      } else {
        cond_.wait_for(lock, time_threshold_,
                       [this, local_gen]() { return local_gen != generation_; });
        if (excp_ptr_) {
          std::rethrow_exception(excp_ptr_);
        }
        if (local_gen == generation_) {
          throw std::runtime_error(ErrorBase + "BlockingCallOnce time out.");
        }
      }
    }

   private:
    std::mutex mu_;
    std::condition_variable cond_;
    const size_t thread_count_;
    volatile size_t count_;
    volatile size_t generation_;
    std::exception_ptr excp_ptr_ = nullptr;
    const std::chrono::seconds time_threshold_{10};

    void* once_callable_;
    void (BlockingCallOnce::*once_call_)();
    template <typename Callable>
    void once_call_impl_() {
      (*(Callable*)once_callable_)();
    }
  };

 public:
  ~CpuResource() = default;

  static std::shared_ptr<CpuResource> Create(const size_t local_gpu_cnt,
                                             const size_t global_gpu_cnt);

  void sync_cpu_threads() const;

  template <typename Callable, typename... Args>
  void blocking_call_once(Callable&& func, Args&&... args) {
    (*blocking_call_oncer_)(std::forward<Callable>(func), std::forward<Args>(args)...);
  }

  template <typename Callable, typename... Args>
  void one_at_a_time(Callable&& func, Args&&... args) {
    std::lock_guard<std::mutex> lock(mu_);
    auto function = std::bind(std::forward<Callable>(func), std::forward<Args>(args)...);
    function();
  }

  template <typename Callable, typename... Args>
  void push_to_threadpool(Callable&& func, Args&&... args) {
    std::function<void()> fn = std::bind(std::forward<Callable>(func), std::forward<Args>(args)...);
    thread_pool_->Schedule(fn);
  }

  template <typename Callable, typename... Args>
  void push_to_workers(const size_t local_replica_id, Callable&& func, Args&&... args) {
    if (local_replica_id >= local_gpu_count_)
      throw std::runtime_error(ErrorBase + "Invalid local_replica_id");
    std::function<void()> fn = std::bind(std::forward<Callable>(func), std::forward<Args>(args)...);
    workers_[local_replica_id]->Schedule(fn);
  }

  void sync_threadpool() const;

  void sync_all_workers_via_mpi() const;

 private:
  CpuResource(const size_t local_gpu_cnt, const size_t global_gpu_cnt);

  const size_t local_gpu_count_;
  const size_t global_gpu_count_;

  std::shared_ptr<Barrier> barrier_;
  std::shared_ptr<BlockingCallOnce> blocking_call_oncer_;
  std::mutex mu_;
  std::unique_ptr<Eigen::SimpleThreadPool> thread_pool_;
  // each GPU has a dedicated threadpool for launching kernels
  std::vector<std::unique_ptr<Eigen::SimpleThreadPool>> workers_;
  MPIContext_t mpi_context_;
};

}  // namespace SparseOperationKit

#endif  // CPU_RESOURCE_H