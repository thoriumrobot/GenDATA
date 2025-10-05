// Source-based slice around line 699
// Method: <com.google.common.util.concurrent.Monitor: boolean enterIf(Guard,Duration)>

  }

  /**
   * Enters this monitor if the guard is satisfied. Blocks at most the given time acquiring the
   * lock, but does not wait for the guard to be satisfied.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean enterIf(Guard guard, Duration time) {
    return enterIf(guard, toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Enters this monitor if the guard is satisfied. Blocks at most the given time acquiring the
   * lock, but does not wait for the guard to be satisfied.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
