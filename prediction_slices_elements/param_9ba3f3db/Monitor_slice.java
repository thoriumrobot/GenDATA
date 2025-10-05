// Source-based slice around line 840
// Method: <com.google.common.util.concurrent.Monitor: boolean waitFor(Guard,Duration)>


  /**
   * Waits for the guard to be satisfied. Waits at most the given time, and may be interrupted. May
   * be called only by a thread currently occupying this monitor.
   *
   * @return whether the guard is now satisfied
   * @throws InterruptedException if interrupted while waiting
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean waitFor(Guard guard, Duration time) throws InterruptedException {
    return waitFor(guard, toNanosSaturated(time), TimeUnit.NANOSECONDS);
  }

  /**
   * Waits for the guard to be satisfied. Waits at most the given time, and may be interrupted. May
   * be called only by a thread currently occupying this monitor.
   *
   * @return whether the guard is now satisfied
   * @throws InterruptedException if interrupted while waiting
   */
