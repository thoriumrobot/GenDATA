// Source-based slice around line 365
// Method: <com.google.common.util.concurrent.RateLimiter: boolean tryAcquire(int)>

   * Acquires permits from this {@link RateLimiter} if it can be acquired immediately without delay.
   *
   * <p>This method is equivalent to {@code tryAcquire(permits, 0, anyUnit)}.
   *
   * @param permits the number of permits to acquire
   * @return {@code true} if the permits were acquired, {@code false} otherwise
   * @throws IllegalArgumentException if the requested number of permits is negative or zero
   * @since 14.0
   */
  public boolean tryAcquire(int permits) {
    return tryAcquire(permits, 0, MICROSECONDS);
  }

  /**
   * Acquires a permit from this {@link RateLimiter} if it can be acquired immediately without
   * delay.
   *
   * <p>This method is equivalent to {@code tryAcquire(1)}.
   *
   * @return {@code true} if the permit was acquired, {@code false} otherwise
