// Source-based slice around line 135
// Method: <com.google.common.util.concurrent.RateLimiter: RateLimiter create(double,SleepingStopwatch)>

     * T3 at 3 seconds
     *
     * Due to the slight delay of T1, T2 would have to sleep till 2.05 seconds, and T3 would also
     * have to sleep till 3.05 seconds.
     */
    return create(permitsPerSecond, SleepingStopwatch.createFromSystemTimer());
  }

  @VisibleForTesting
  static RateLimiter create(double permitsPerSecond, SleepingStopwatch stopwatch) {
    RateLimiter rateLimiter = new SmoothBursty(stopwatch, 1.0 /* maxBurstSeconds */);
    rateLimiter.setRate(permitsPerSecond);
    return rateLimiter;
  }

  /**
   * Creates a {@code RateLimiter} with the specified stable throughput, given as "permits per
   * second" (commonly referred to as <i>QPS</i>, queries per second), and a <i>warmup period</i>,
   * during which the {@code RateLimiter} smoothly ramps up its rate, until it reaches its maximum
   * rate at the end of the period (as long as there are enough requests to saturate it). Similarly,
