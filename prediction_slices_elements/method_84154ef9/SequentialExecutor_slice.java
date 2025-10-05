// Source-based slice around line 268
// Method: <com.google.common.util.concurrent.SequentialExecutor: String toString()>

      Runnable currentlyRunning = task;
      if (currentlyRunning != null) {
        return "SequentialExecutorWorker{running=" + currentlyRunning + "}";
      }
      return "SequentialExecutorWorker{state=" + workerRunningState + "}";
    }
  }

  @Override
  public String toString() {
    return "SequentialExecutor@" + identityHashCode(this) + "{" + executor + "}";
  }
}
