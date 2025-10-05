// Source-based slice around line 950
// Method: <com.google.common.util.concurrent.Monitor: boolean isOccupied()>

  /** Returns whether this monitor is using a fair ordering policy. */
  public boolean isFair() {
    return fair;
  }

  /**
   * Returns whether this monitor is occupied by any thread. This method is designed for use in
   * monitoring of the system state, not for synchronization control.
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
