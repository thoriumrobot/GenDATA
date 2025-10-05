// Source-based slice around line 976
// Method: <com.google.common.util.concurrent.Monitor: int getQueueLength()>

    return lock.getHoldCount();
  }

  /**
   * Returns an estimate of the number of threads waiting to enter this monitor. The value is only
   * an estimate because the number of threads may change dynamically while this method traverses
   * internal data structures. This method is designed for use in monitoring of the system state,
   * not for synchronization control.
   */
  public int getQueueLength() {
    return lock.getQueueLength();
  }

  /**
   * Returns whether any threads are waiting to enter this monitor. Note that because cancellations
   * may occur at any time, a {@code true} return does not guarantee that any other thread will ever
   * enter this monitor. This method is designed primarily for use in monitoring of the system
   * state.
   */
  public boolean hasQueuedThreads() {
