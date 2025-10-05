// Source-based slice around line 77
// Method: <com.google.common.util.concurrent.DirectExecutorService: List shutdownNow()>

      shutdown = true;
      if (runningTasks == 0) {
        lock.notifyAll();
      }
    }
  }

  // See newDirectExecutorService javadoc for unusual behavior of this method.
  @Override
  public List<Runnable> shutdownNow() {
    shutdown();
    return ImmutableList.of();
  }

  @Override
  public boolean isTerminated() {
    synchronized (lock) {
      return shutdown && runningTasks == 0;
    }
  }
