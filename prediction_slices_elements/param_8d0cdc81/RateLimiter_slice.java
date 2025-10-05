// Source-based slice around line 202
// Method: <com.google.common.util.concurrent.RateLimiter: RateLimiter create(double,long,TimeUnit,double,SleepingStopwatch)>

  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public static RateLimiter create(double permitsPerSecond, long warmupPeriod, TimeUnit unit) {
    checkArgument(warmupPeriod >= 0, "warmupPeriod must not be negative: %s", warmupPeriod);
    return create(
        permitsPerSecond, warmupPeriod, unit, 3.0, SleepingStopwatch.createFromSystemTimer());
  }

  @VisibleForTesting
  static RateLimiter create(
      double permitsPerSecond,
      long warmupPeriod,
      TimeUnit unit,
      double coldFactor,
      SleepingStopwatch stopwatch) {
    RateLimiter rateLimiter = new SmoothWarmingUp(stopwatch, warmupPeriod, unit, coldFactor);
    rateLimiter.setRate(permitsPerSecond);
    return rateLimiter;
  }

  /**
