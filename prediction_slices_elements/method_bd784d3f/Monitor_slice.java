// Source-based slice around line 958
// Method: <com.google.common.util.concurrent.Monitor: boolean isOccupiedByCurrentThread()>

   */
  public boolean isOccupied() {
    return lock.isLocked();
  }

  /**
   * Returns whether the current thread is occupying this monitor (has entered more times than it
   * has left).
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
