// Source-based slice around line 996
// Method: <com.google.common.util.concurrent.Monitor: boolean hasQueuedThread(Thread)>

    return lock.hasQueuedThreads();
  }

  /**
   * Queries whether the given thread is waiting to enter this monitor. Note that because
   * cancellations may occur at any time, a {@code true} return does not guarantee that this thread
   * will ever enter this monitor. This method is designed primarily for use in monitoring of the
   * system state.
   */
  public boolean hasQueuedThread(Thread thread) {
    return lock.hasQueuedThread(thread);
  }

  /**
   * Queries whether any threads are waiting for the given guard to become satisfied. Note that
   * because timeouts and interrupts may occur at any time, a {@code true} return does not guarantee
   * that the guard becoming satisfied in the future will awaken any threads. This method is
   * designed primarily for use in monitoring of the system state.
   */
  public boolean hasWaiters(Guard guard) {
