// Source-based slice around line 735
// Method: <com.google.common.util.concurrent.Monitor: boolean enterIfInterruptibly(Guard)>

  }

  /**
   * Enters this monitor if the guard is satisfied. Blocks indefinitely acquiring the lock, but does
   * not wait for the guard to be satisfied, and may be interrupted.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   * @throws InterruptedException if interrupted while waiting
   */
  public boolean enterIfInterruptibly(Guard guard) throws InterruptedException {
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    lock.lockInterruptibly();

    boolean satisfied = false;
    try {
      return satisfied = guard.isSatisfied();
    } finally {
