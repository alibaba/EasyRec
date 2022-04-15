// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_SIMPLE_THREAD_POOL_H
#define EIGEN_CXX11_THREADPOOL_SIMPLE_THREAD_POOL_H

#include "ThreadPoolInterface.h"
#include "ThreadEnvironment.h"
#include <vector>
#include <deque>
#include <condition_variable>
#include <stdexcept>

namespace Eigen {

// The implementation of the ThreadPool type ensures that the Schedule method
// runs the functions it is provided in FIFO order when the scheduling is done
// by a single thread.
// Environment provides a way to create threads and also allows to intercept
// task submission and execution.
template <typename Environment>
class SimpleThreadPoolTempl : public ThreadPoolInterface {
 public:
  // Construct a pool that contains "num_threads" threads.
  explicit SimpleThreadPoolTempl(const int num_threads, Environment env = Environment())
      : env_(env), num_threads_(num_threads), waiters_(0)
  {
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(env.CreateThread([this, i]() { WorkerLoop(i); }));
    }
  }

  // Wait until all scheduled work has finished and then destroy the
  // set of threads.
  ~SimpleThreadPoolTempl() {
    {
      // Wait for all work to get done.
      std::unique_lock<std::mutex> l(mu_);
      while (!pending_.empty()) {
        empty_.wait(l);
      }
      exiting_ = true;

      // Wakeup all waiters.
      for (auto w : waiters_) {
        w->ready = true;
        w->task.f = nullptr;
        w->cv.notify_one();
      }
    }

    // Wait for threads to finish.
    for (auto t : threads_) {
      delete t;
    }
  }

  bool Done() const {
    return num_threads_ == static_cast<int>(waiters_.size());
  }

  // Schedule fn() for execution in the pool of threads. The functions are
  // executed in the order in which they are scheduled.
  void Schedule(std::function<void()> fn) final {
    Task t = env_.CreateTask(std::move(fn));
    std::unique_lock<std::mutex> l(mu_);
    if (waiters_.empty()) {
      pending_.push_back(std::move(t));
    } else {
      Waiter* w = waiters_.back();
      waiters_.pop_back();
      w->ready = true;
      w->task = std::move(t);
      w->cv.notify_one();
    }
  }

  int NumThreads() const final {
    return static_cast<int>(threads_.size());
  }

  int CurrentThreadId() const final {
    const PerThread* pt = this->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

 protected:
  void WorkerLoop(int thread_id) {
    try {
      std::unique_lock<std::mutex> l(mu_);
      PerThread* pt = GetPerThread();
      pt->pool = this;
      pt->thread_id = thread_id;
      Waiter w;
      Task t;
      while (!exiting_) {
        if (pending_.empty()) {
          // Wait for work to be assigned to me
          w.ready = false;
          waiters_.push_back(&w);
          while (!w.ready) {
            w.cv.wait(l);
          }
          t = w.task;
          w.task.f = nullptr;
        } else {
          // Pick up pending work
          if (exiting_ && pending_.empty()) return;
          if (exce_ptr_ != nullptr) std::rethrow_exception(exce_ptr_);
          
          t = std::move(pending_.front());
          pending_.pop_front();
          if (pending_.empty()) {
            empty_.notify_all();
          }
        }
        if (t.f) {
          mu_.unlock();
          env_.ExecuteTask(t);
          t.f = nullptr;
          mu_.lock();
        }
      }
    } catch (...) {
      exiting_ = true;
      empty_.notify_all();
      exce_ptr_ = std::current_exception();
      std::rethrow_exception(exce_ptr_);
    }
  }

 private:
  typedef typename Environment::Task Task;
  typedef typename Environment::EnvThread Thread;

  struct Waiter {
    std::condition_variable cv;
    Task task;
    bool ready;
  };

  struct PerThread {
    constexpr PerThread() : pool(NULL), thread_id(-1) { }
    SimpleThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    int thread_id;                // Worker thread index in pool.
  };

  Environment env_;
  std::mutex mu_;
  // MaxSizeVector<Thread*> threads_;  // All threads
  // MaxSizeVector<Waiter*> waiters_;  // Stack of waiting threads.
  std::vector<Thread*> threads_; // All threads
  const int num_threads_; // num_threads
  std::vector<Waiter*> waiters_; // Stack of waiting threads.
  std::deque<Task> pending_;        // Queue of pending work
  std::condition_variable empty_;   // Signaled on pending_.empty()
  bool exiting_ = false;
  std::exception_ptr exce_ptr_;

  PerThread* GetPerThread() const {
    static thread_local PerThread per_thread;
    return &per_thread;
  }
};

typedef SimpleThreadPoolTempl<StlThreadEnvironment> SimpleThreadPool;

}  // namespace Eigen

#endif  // EIGEN_CXX11_THREADPOOL_SIMPLE_THREAD_POOL_H
