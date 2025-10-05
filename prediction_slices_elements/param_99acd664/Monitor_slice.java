// Source-based slice around line 886
// Method: <com.google.common.util.concurrent.Monitor: boolean waitForUninterruptibly(Guard,Duration)>

  }

  /**
   * Waits for the guard to be satisfied. Waits at most the given time. May be called only by a
   * thread currently occupying this monitor.
   *
   * @return whether the guard is now satisfied
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean waitForUninterruptibly(Guard guard, Duration time) {
    return waitForUninterruptibly(guard, toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Waits for the guard to be satisfied. Waits at most the given time. May be called only by a
   * thread currently occupying this monitor.
   *
   * @return whether the guard is now satisfied
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
