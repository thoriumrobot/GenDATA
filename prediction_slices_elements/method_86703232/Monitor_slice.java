// Source-based slice around line 966
// Method: <com.google.common.util.concurrent.Monitor: int getOccupiedDepth()>

   */
  public boolean isOccupiedByCurrentThread() {
    return lock.isHeldByCurrentThread();
  }

  /**
   * Returns the number of times the current thread has entered this monitor in excess of the number
   * of times it has left. Returns 0 if the current thread is not occupying this monitor.
   */
  public int getOccupiedDepth() {
    return lock.getHoldCount();
  }

  /**
   * Returns an estimate of the number of threads waiting to enter this monitor. The value is only
   * an estimate because the number of threads may change dynamically while this method traverses
   * internal data structures. This method is designed for use in monitoring of the system state,
   * not for synchronization control.
   */
  public int getQueueLength() {
