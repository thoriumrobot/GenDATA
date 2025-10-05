// Source-based slice around line 289
// Method: <com.google.common.util.concurrent.RateLimiter: double acquire()>

   * Acquires a single permit from this {@code RateLimiter}, blocking until the request can be
   * granted. Tells the amount of time slept, if any.
   *
   * <p>This method is equivalent to {@code acquire(1)}.
   *
   * @return time spent sleeping to enforce rate, in seconds; 0.0 if not rate-limited
   * @since 16.0 (present in 13.0 with {@code void} return type})
   */
  @CanIgnoreReturnValue
  public double acquire() {
    return acquire(1);
  }

  /**
   * Acquires the given number of permits from this {@code RateLimiter}, blocking until the request
   * can be granted. Tells the amount of time slept, if any.
   *
   * @param permits the number of permits to acquire
   * @return time spent sleeping to enforce rate, in seconds; 0.0 if not rate-limited
   * @throws IllegalArgumentException if the requested number of permits is negative or zero
