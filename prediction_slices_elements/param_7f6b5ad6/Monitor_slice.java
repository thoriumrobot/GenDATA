// Source-based slice around line 1006
// Method: <com.google.common.util.concurrent.Monitor: boolean hasWaiters(Guard)>

    return lock.hasQueuedThread(thread);
  }

  /**
   * Queries whether any threads are waiting for the given guard to become satisfied. Note that
   * because timeouts and interrupts may occur at any time, a {@code true} return does not guarantee
   * that the guard becoming satisfied in the future will awaken any threads. This method is
   * designed primarily for use in monitoring of the system state.
   */
  public boolean hasWaiters(Guard guard) {
    return getWaitQueueLength(guard) > 0;
  }

  /**
   * Returns an estimate of the number of threads waiting for the given guard to become satisfied.
   * Note that because timeouts and interrupts may occur at any time, the estimate serves only as an
   * upper bound on the actual number of waiters. This method is designed for use in monitoring of
   * the system state, not for synchronization control.
   */
  public int getWaitQueueLength(Guard guard) {
