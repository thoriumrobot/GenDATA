// Source-based slice around line 609
// Method: <com.google.common.util.concurrent.Monitor: boolean enterWhenUninterruptibly(Guard,long,TimeUnit)>

  }

  /**
   * Enters this monitor when the guard is satisfied. Blocks at most the given time, including both
   * the time to acquire the lock and the time to wait for the guard to be satisfied.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean enterWhenUninterruptibly(Guard guard, long time, TimeUnit unit) {
    long timeoutNanos = toSafeNanos(time, unit);
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    long startTime = 0L;
    boolean signalBeforeWaiting = lock.isHeldByCurrentThread();
    boolean interrupted = Thread.interrupted();
    try {
      if (fair || !lock.tryLock()) {
