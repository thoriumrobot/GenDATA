// Source-based slice around line 277
// Method: <com.google.common.util.concurrent.RateLimiter: double doGetRate()>

   * passed in the factory method that produced this {@code RateLimiter}, and it is only updated
   * after invocations to {@linkplain #setRate}.
   */
  public final double getRate() {
    synchronized (mutex()) {
      return doGetRate();
    }
  }

  abstract double doGetRate();

  /**
   * Acquires a single permit from this {@code RateLimiter}, blocking until the request can be
   * granted. Tells the amount of time slept, if any.
   *
   * <p>This method is equivalent to {@code acquire(1)}.
   *
   * @return time spent sleeping to enforce rate, in seconds; 0.0 if not rate-limited
   * @since 16.0 (present in 13.0 with {@code void} return type})
   */
