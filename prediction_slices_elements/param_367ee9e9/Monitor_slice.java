// Source-based slice around line 502
// Method: <com.google.common.util.concurrent.Monitor: boolean enterWhen(Guard,Duration)>

  /**
   * Enters this monitor when the guard is satisfied. Blocks at most the given time, including both
   * the time to acquire the lock and the time to wait for the guard to be satisfied, and may be
   * interrupted.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   * @throws InterruptedException if interrupted while waiting
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean enterWhen(Guard guard, Duration time) throws InterruptedException {
    return enterWhen(guard, toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Enters this monitor when the guard is satisfied. Blocks at most the given time, including both
   * the time to acquire the lock and the time to wait for the guard to be satisfied, and may be
   * interrupted.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   * @throws InterruptedException if interrupted while waiting
