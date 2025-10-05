// Source-based slice around line 117
// Method: <com.google.common.util.concurrent.RateLimiter: RateLimiter create(double)>

   * rate limiter is unused, bursts of up to {@code permitsPerSecond} permits will be allowed, with
   * subsequent requests being smoothly limited at the stable rate of {@code permitsPerSecond}.
   *
   * @param permitsPerSecond the rate of the returned {@code RateLimiter}, measured in how many
   *     permits become available per second
   * @throws IllegalArgumentException if {@code permitsPerSecond} is negative or zero
   */
  // TODO(user): "This is equivalent to
  // {@code createWithCapacity(permitsPerSecond, 1, TimeUnit.SECONDS)}".
  public static RateLimiter create(double permitsPerSecond) {
    /*
     * The default RateLimiter configuration can save the unused permits of up to one second. This
     * is to avoid unnecessary stalls in situations like this: A RateLimiter of 1qps, and 4 threads,
     * all calling acquire() at these moments:
     *
     * T0 at 0 seconds
     * T1 at 1.05 seconds
     * T2 at 2 seconds
     * T3 at 3 seconds
     *
