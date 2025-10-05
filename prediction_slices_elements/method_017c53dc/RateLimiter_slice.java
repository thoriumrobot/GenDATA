// Source-based slice around line 303
// Method: <com.google.common.util.concurrent.RateLimiter: double acquire(int)>

   * Acquires the given number of permits from this {@code RateLimiter}, blocking until the request
   * can be granted. Tells the amount of time slept, if any.
   *
   * @param permits the number of permits to acquire
   * @return time spent sleeping to enforce rate, in seconds; 0.0 if not rate-limited
   * @throws IllegalArgumentException if the requested number of permits is negative or zero
   * @since 16.0 (present in 13.0 with {@code void} return type})
   */
  @CanIgnoreReturnValue
  public double acquire(int permits) {
    long microsToWait = reserve(permits);
    stopwatch.sleepMicrosUninterruptibly(microsToWait);
    return 1.0 * microsToWait / SECONDS.toMicros(1L);
  }

  /**
   * Reserves the given number of permits from this {@code RateLimiter} for future use, returning
   * the number of microseconds until the reservation can be consumed.
   *
   * @return time in microseconds to wait until the resource can be acquired, never negative
