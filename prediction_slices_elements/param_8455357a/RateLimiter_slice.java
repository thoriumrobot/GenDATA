// Source-based slice around line 351
// Method: <com.google.common.util.concurrent.RateLimiter: boolean tryAcquire(long,TimeUnit)>

   *
   * <p>This method is equivalent to {@code tryAcquire(1, timeout, unit)}.
   *
   * @param timeout the maximum time to wait for the permit. Negative values are treated as zero.
   * @param unit the time unit of the timeout argument
   * @return {@code true} if the permit was acquired, {@code false} otherwise
   * @throws IllegalArgumentException if the requested number of permits is negative or zero
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean tryAcquire(long timeout, TimeUnit unit) {
    return tryAcquire(1, timeout, unit);
  }

  /**
   * Acquires permits from this {@link RateLimiter} if it can be acquired immediately without delay.
   *
   * <p>This method is equivalent to {@code tryAcquire(permits, 0, anyUnit)}.
   *
   * @param permits the number of permits to acquire
   * @return {@code true} if the permits were acquired, {@code false} otherwise
