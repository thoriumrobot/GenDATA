// Source-based slice around line 194
// Method: <com.google.common.util.concurrent.RateLimiter: RateLimiter create(double,long,TimeUnit)>

   * @param permitsPerSecond the rate of the returned {@code RateLimiter}, measured in how many
   *     permits become available per second
   * @param warmupPeriod the duration of the period where the {@code RateLimiter} ramps up its rate,
   *     before reaching its stable (maximum) rate
   * @param unit the time unit of the warmupPeriod argument
   * @throws IllegalArgumentException if {@code permitsPerSecond} is negative or zero or {@code
   *     warmupPeriod} is negative
   */
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
