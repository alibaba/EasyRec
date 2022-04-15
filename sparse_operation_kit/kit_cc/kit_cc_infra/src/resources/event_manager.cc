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

#include "resources/event_manager.h"

#include "common.h"

namespace SparseOperationKit {

EventManager::EventManager() : shared_mu_() {}

EventManager::~EventManager() {}

std::unique_ptr<EventManager> EventManager::create() {
  return std::unique_ptr<EventManager>(new EventManager());
}

std::shared_ptr<Event>& EventManager::create_event(const std::string event_name) {
  std::unique_lock<std::shared_timed_mutex> unique(shared_mu_);
  auto iter = events_.find(event_name);
  if (events_.end() == iter) {
    std::shared_ptr<Event> event = Event::create(event_name);
    events_.emplace(std::make_pair(event_name, event));
    iter = events_.find(event_name);
  }
  return iter->second;
}

std::shared_ptr<Event>& EventManager::get_event(const std::string event_name) {
  {
    std::shared_lock<std::shared_timed_mutex> shared(shared_mu_);
    auto iter = events_.find(event_name);
    if (events_.end() != iter) return iter->second;
  }
  return create_event(std::move(event_name));  // no such event
}

void EventManager::sync_two_streams(cudaStream_t& root_stream, cudaStream_t& sub_stream,
                                    const std::string event_name, const bool event_sync) {
  /*--root_stream->event->sub_stream--*/
  std::shared_ptr<Event>& event = get_event(std::move(event_name));
  event->Record(root_stream);
  return event_sync ? event->TillReady() : event->TillReady(sub_stream);
}

}  // namespace SparseOperationKit