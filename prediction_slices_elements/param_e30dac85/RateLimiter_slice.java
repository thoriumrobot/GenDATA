// Source-based slice around line 393
// Method: <com.google.common.util.concurrent.RateLimiter: boolean tryAcquire(int,Duration)>

   * without exceeding the specified {@code timeout}, or returns {@code false} immediately (without
   * waiting) if the permits would not have been granted before the timeout expired.
   *
   * @param permits the number of permits to acquire
   * @param timeout the maximum time to wait for the permits. Negative values are treated as zero.
   * @return {@code true} if the permits were acquired, {@code false} otherwise
   * @throws IllegalArgumentException if the requested number of permits is negative or zero
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean tryAcquire(int permits, Duration timeout) {
    return tryAcquire(permits, toNanosSaturated(timeout), TimeUnit.NANOSECONDS);
  }

  /**
   * Acquires the given number of permits from this {@code RateLimiter} if it can be obtained
   * without exceeding the specified {@code timeout}, or returns {@code false} immediately (without
   * waiting) if the permits would not have been granted before the timeout expired.
   *
   * @param permits the number of permits to acquire
   * @param timeout the maximum time to wait for the permits. Negative values are treated as zero.
