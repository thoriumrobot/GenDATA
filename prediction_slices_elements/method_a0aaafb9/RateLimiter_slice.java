// Source-based slice around line 378
// Method: <com.google.common.util.concurrent.RateLimiter: boolean tryAcquire()>

  /**
   * Acquires a permit from this {@link RateLimiter} if it can be acquired immediately without
   * delay.
   *
   * <p>This method is equivalent to {@code tryAcquire(1)}.
   *
   * @return {@code true} if the permit was acquired, {@code false} otherwise
   * @since 14.0
   */
  public boolean tryAcquire() {
    return tryAcquire(1, 0, MICROSECONDS);
  }

  /**
   * Acquires the given number of permits from this {@code RateLimiter} if it can be obtained
   * without exceeding the specified {@code timeout}, or returns {@code false} immediately (without
   * waiting) if the permits would not have been granted before the timeout expired.
   *
   * @param permits the number of permits to acquire
   * @param timeout the maximum time to wait for the permits. Negative values are treated as zero.
