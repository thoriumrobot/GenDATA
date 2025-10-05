// Source-based slice around line 122
// Method: <com.google.common.util.concurrent.DirectExecutorService: void endTask()>

    synchronized (lock) {
      if (shutdown) {
        throw new RejectedExecutionException("Executor already shutdown");
      }
      runningTasks++;
    }
  }

  /** Decrements the running task count. */
  private void endTask() {
    synchronized (lock) {
      int numRunning = --runningTasks;
      if (numRunning == 0) {
        lock.notifyAll();
      }
    }
  }
}
