// Source-based slice around line 112
// Method: <com.google.common.util.concurrent.DirectExecutorService: void startTask()>

      }
    }
  }

  /**
   * Checks if the executor has been shut down and increments the running task count.
   *
   * @throws RejectedExecutionException if the executor has been previously shutdown
   */
  private void startTask() {
    synchronized (lock) {
      if (shutdown) {
        throw new RejectedExecutionException("Executor already shutdown");
      }
      runningTasks++;
    }
  }

  /** Decrements the running task count. */
  private void endTask() {
