// Source-based slice around line 165
// Method: <com.google.common.util.concurrent.RateLimiter: RateLimiter create(double,Duration)>

   *
   * @param permitsPerSecond the rate of the returned {@code RateLimiter}, measured in how many
   *     permits become available per second
   * @param warmupPeriod the duration of the period where the {@code RateLimiter} ramps up its rate,
   *     before reaching its stable (maximum) rate
   * @throws IllegalArgumentException if {@code permitsPerSecond} is negative or zero or {@code
   *     warmupPeriod} is negative
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public static RateLimiter create(double permitsPerSecond, Duration warmupPeriod) {
    return create(permitsPerSecond, toNanosSaturated(warmupPeriod), TimeUnit.NANOSECONDS);
  }

  /**
   * Creates a {@code RateLimiter} with the specified stable throughput, given as "permits per
   * second" (commonly referred to as <i>QPS</i>, queries per second), and a <i>warmup period</i>,
   * during which the {@code RateLimiter} smoothly ramps up its rate, until it reaches its maximum
   * rate at the end of the period (as long as there are enough requests to saturate it). Similarly,
   * if the {@code RateLimiter} is left <i>unused</i> for a duration of {@code warmupPeriod}, it
   * will gradually return to its "cold" state, i.e. it will go through the same warming up process
