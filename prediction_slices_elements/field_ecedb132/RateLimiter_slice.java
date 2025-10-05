// Source-based slice around line 216
// Method: com.google.common.util.concurrent.RateLimiter.stopwatch

    RateLimiter rateLimiter = new SmoothWarmingUp(stopwatch, warmupPeriod, unit, coldFactor);
    rateLimiter.setRate(permitsPerSecond);
    return rateLimiter;
  }

  /**
   * The underlying timer; used both to measure elapsed time and sleep as necessary. A separate
   * object to facilitate testing.
   */
  private final SleepingStopwatch stopwatch;

  // Can't be initialized in the constructor because mocks don't call the constructor.
  private volatile @Nullable Object mutexDoNotUseDirectly;

  private Object mutex() {
    Object mutex = mutexDoNotUseDirectly;
    if (mutex == null) {
      synchronized (this) {
        mutex = mutexDoNotUseDirectly;
        if (mutex == null) {
