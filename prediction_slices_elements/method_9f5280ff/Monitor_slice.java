// Source-based slice around line 391
// Method: <com.google.common.util.concurrent.Monitor: boolean enter(Duration)>

    lock.lock();
  }

  /**
   * Enters this monitor. Blocks at most the given time.
   *
   * @return whether the monitor was entered
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean enter(Duration time) {
    return enter(toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Enters this monitor. Blocks at most the given time.
   *
   * @return whether the monitor was entered
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean enter(long time, TimeUnit unit) {
