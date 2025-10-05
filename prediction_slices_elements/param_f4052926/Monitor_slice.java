// Source-based slice around line 401
// Method: <com.google.common.util.concurrent.Monitor: boolean enter(long,TimeUnit)>

    return enter(toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Enters this monitor. Blocks at most the given time.
   *
   * @return whether the monitor was entered
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean enter(long time, TimeUnit unit) {
    long timeoutNanos = toSafeNanos(time, unit);
    ReentrantLock lock = this.lock;
    if (!fair && lock.tryLock()) {
      return true;
    }
    boolean interrupted = Thread.interrupted();
    try {
      long startTime = System.nanoTime();
      for (long remainingNanos = timeoutNanos; ; ) {
        try {
