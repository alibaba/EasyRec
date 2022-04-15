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

#ifndef RESOURCES_EVENT_H
#define RESOURCES_EVENT_H

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

namespace SparseOperationKit {

/*type used to specify the event is recorded on which stream*/
enum class EventRecordType { RDLFramework, RMyself };

/*class used to handle cudaEvent*/
class Event {
 public:
  static std::shared_ptr<Event> create(const std::string name);
  ~Event();

  Event(Event& event) = delete;
  Event& operator=(Event& event) = delete;
  Event(Event&& event) = delete;
  Event& operator=(Event&& event) = delete;

  void Record(cudaStream_t& stream);
  bool IsReady() const;
  void TillReady(cudaStream_t& stream);
  void TillReady();
  std::string name() const;

 protected:
  explicit Event(const std::string name);

 private:
  cudaEvent_t cuda_event_{nullptr};
  const std::string name_;
};

}  // namespace SparseOperationKit

#endif  // RESOURCES_EVENT_H