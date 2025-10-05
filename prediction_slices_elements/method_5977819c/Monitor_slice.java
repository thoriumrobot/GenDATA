// Source-based slice around line 986
// Method: <com.google.common.util.concurrent.Monitor: boolean hasQueuedThreads()>

    return lock.getQueueLength();
  }

  /**
   * Returns whether any threads are waiting to enter this monitor. Note that because cancellations
   * may occur at any time, a {@code true} return does not guarantee that any other thread will ever
   * enter this monitor. This method is designed primarily for use in monitoring of the system
   * state.
   */
  public boolean hasQueuedThreads() {
    return lock.hasQueuedThreads();
  }

  /**
   * Queries whether the given thread is waiting to enter this monitor. Note that because
   * cancellations may occur at any time, a {@code true} return does not guarantee that this thread
   * will ever enter this monitor. This method is designed primarily for use in monitoring of the
   * system state.
   */
  public boolean hasQueuedThread(Thread thread) {
