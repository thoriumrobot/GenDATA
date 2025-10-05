// Source-based slice around line 452
// Method: <com.google.common.util.concurrent.Monitor: boolean enterInterruptibly(long,TimeUnit)>

  }

  /**
   * Enters this monitor. Blocks at most the given time, and may be interrupted.
   *
   * @return whether the monitor was entered
   * @throws InterruptedException if interrupted while waiting
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean enterInterruptibly(long time, TimeUnit unit) throws InterruptedException {
    return lock.tryLock(time, unit);
  }

  /**
   * Enters this monitor if it is possible to do so immediately. Does not block.
   *
   * <p><b>Note:</b> This method disregards the fairness setting of this monitor.
   *
   * @return whether the monitor was entered
   */
