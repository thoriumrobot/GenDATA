// Source-based slice around line 1016
// Method: <com.google.common.util.concurrent.Monitor: int getWaitQueueLength(Guard)>

    return getWaitQueueLength(guard) > 0;
  }

  /**
   * Returns an estimate of the number of threads waiting for the given guard to become satisfied.
   * Note that because timeouts and interrupts may occur at any time, the estimate serves only as an
   * upper bound on the actual number of waiters. This method is designed for use in monitoring of
   * the system state, not for synchronization control.
   */
  public int getWaitQueueLength(Guard guard) {
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    lock.lock();
    try {
      return guard.waiterCount;
    } finally {
      lock.unlock();
    }
  }
