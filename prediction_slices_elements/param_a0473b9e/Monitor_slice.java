// Source-based slice around line 598
// Method: <com.google.common.util.concurrent.Monitor: boolean enterWhenUninterruptibly(Guard,Duration)>

  }

  /**
   * Enters this monitor when the guard is satisfied. Blocks at most the given time, including both
   * the time to acquire the lock and the time to wait for the guard to be satisfied.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean enterWhenUninterruptibly(Guard guard, Duration time) {
    return enterWhenUninterruptibly(guard, toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Enters this monitor when the guard is satisfied. Blocks at most the given time, including both
   * the time to acquire the lock and the time to wait for the guard to be satisfied.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
