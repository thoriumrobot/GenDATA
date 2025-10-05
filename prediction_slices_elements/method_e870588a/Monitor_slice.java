// Source-based slice around line 798
// Method: <com.google.common.util.concurrent.Monitor: boolean tryEnterIf(Guard)>


  /**
   * Enters this monitor if it is possible to do so immediately and the guard is satisfied. Does not
   * block acquiring the lock and does not wait for the guard to be satisfied.
   *
   * <p><b>Note:</b> This method disregards the fairness setting of this monitor.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   */
  public boolean tryEnterIf(Guard guard) {
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    if (!lock.tryLock()) {
      return false;
    }

    boolean satisfied = false;
    try {
