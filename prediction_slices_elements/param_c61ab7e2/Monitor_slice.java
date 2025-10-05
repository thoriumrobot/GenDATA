// Source-based slice around line 441
// Method: <com.google.common.util.concurrent.Monitor: boolean enterInterruptibly(Duration)>

  }

  /**
   * Enters this monitor. Blocks at most the given time, and may be interrupted.
   *
   * @return whether the monitor was entered
   * @throws InterruptedException if interrupted while waiting
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean enterInterruptibly(Duration time) throws InterruptedException {
    return enterInterruptibly(toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Enters this monitor. Blocks at most the given time, and may be interrupted.
   *
   * @return whether the monitor was entered
   * @throws InterruptedException if interrupted while waiting
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
