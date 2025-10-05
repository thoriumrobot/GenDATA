// Source-based slice around line 675
// Method: <com.google.common.util.concurrent.Monitor: boolean enterIf(Guard)>

    }
  }

  /**
   * Enters this monitor if the guard is satisfied. Blocks indefinitely acquiring the lock, but does
   * not wait for the guard to be satisfied.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   */
  public boolean enterIf(Guard guard) {
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    lock.lock();

    boolean satisfied = false;
    try {
      return satisfied = guard.isSatisfied();
    } finally {
