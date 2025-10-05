// Source-based slice around line 334
// Method: <com.google.common.util.concurrent.RateLimiter: boolean tryAcquire(Duration)>

   * would not have been granted before the timeout expired.
   *
   * <p>This method is equivalent to {@code tryAcquire(1, timeout)}.
   *
   * @param timeout the maximum time to wait for the permit. Negative values are treated as zero.
   * @return {@code true} if the permit was acquired, {@code false} otherwise
   * @throws IllegalArgumentException if the requested number of permits is negative or zero
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean tryAcquire(Duration timeout) {
    return tryAcquire(1, toNanosSaturated(timeout), TimeUnit.NANOSECONDS);
  }

  /**
   * Acquires a permit from this {@code RateLimiter} if it can be obtained without exceeding the
   * specified {@code timeout}, or returns {@code false} immediately (without waiting) if the permit
   * would not have been granted before the timeout expired.
   *
   * <p>This method is equivalent to {@code tryAcquire(1, timeout, unit)}.
   *
