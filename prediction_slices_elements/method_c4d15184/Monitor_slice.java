// Source-based slice around line 770
// Method: <com.google.common.util.concurrent.Monitor: boolean enterIfInterruptibly(Guard,long,TimeUnit)>

  }

  /**
   * Enters this monitor if the guard is satisfied. Blocks at most the given time acquiring the
   * lock, but does not wait for the guard to be satisfied, and may be interrupted.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean enterIfInterruptibly(Guard guard, long time, TimeUnit unit)
      throws InterruptedException {
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    if (!lock.tryLock(time, unit)) {
      return false;
    }

    boolean satisfied = false;
