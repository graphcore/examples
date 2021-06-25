// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <thread>
#include "ipu_utils.hpp"

/// Class for launching asynchronous processing tasks
/// in a separate thread. Tasks that are launched must
/// be stand alone and eventually terminate of their
/// own accord.
class AsyncTask {
public:
  AsyncTask() {}
  virtual ~AsyncTask() {}

  /// Run a thread forwarding all arguments to thread constructor.
  /// The thread must terminate of its own accord.
  /// Throws std::logic_error if a task was already in progress.
  template <typename ...Args>
  void run(Args&& ...args) {
    if (job != nullptr) {
      auto error = "Attempted to run AsyncTask while a job was in progress.";
      utils::logger()->error(error);
      throw std::logic_error(error);
    }
    job.reset(new std::thread(std::forward<Args>(args)...));
  }

  /// Wait for the job to complete. Throws std::system_error if
  /// the thread could not be joined.
  void waitForCompletion() {
    if (job != nullptr) {
      try {
        job->join();
        job.reset();
      } catch (std::system_error& e) {
        utils::logger()->error("Thread could not be joined.");
      }
    }
  }

private:
    std::unique_ptr<std::thread> job;
};
