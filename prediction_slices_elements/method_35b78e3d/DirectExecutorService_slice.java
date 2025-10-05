// Source-based slice around line 66
// Method: <com.google.common.util.concurrent.DirectExecutorService: void shutdown()>


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
        lock.notifyAll();
      }
    }
  }

  // See newDirectExecutorService javadoc for unusual behavior of this method.
  @Override
