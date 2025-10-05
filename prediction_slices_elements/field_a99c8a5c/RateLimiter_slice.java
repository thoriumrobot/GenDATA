// Source-based slice around line 219
// Method: com.google.common.util.concurrent.RateLimiter.mutexDoNotUseDirectly

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
          mutexDoNotUseDirectly = mutex = new Object();
        }
      }
