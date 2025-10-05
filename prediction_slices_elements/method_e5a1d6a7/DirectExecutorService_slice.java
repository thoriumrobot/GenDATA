// Source-based slice around line 59
// Method: <com.google.common.util.concurrent.DirectExecutorService: boolean isShutdown()>

    startTask();
    try {
      command.run();
    } finally {
      endTask();
    }
  }

  @Override
  public boolean isShutdown() {
    synchronized (lock) {
      return shutdown;
    }
  }

  @Override
  public void shutdown() {
    synchronized (lock) {
      shutdown = true;
      if (runningTasks == 0) {
