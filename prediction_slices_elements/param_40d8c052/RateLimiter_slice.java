// Source-based slice around line 256
// Method: <com.google.common.util.concurrent.RateLimiter: void setRate(double)>

   * which is in terms of the previous rate.
   *
   * <p>The behavior of the {@code RateLimiter} is not modified in any other way, e.g. if the {@code
   * RateLimiter} was configured with a warmup period of 20 seconds, it still has a warmup period of
   * 20 seconds after this method invocation.
   *
   * @param permitsPerSecond the new stable rate of this {@code RateLimiter}
   * @throws IllegalArgumentException if {@code permitsPerSecond} is negative or zero
   */
  public final void setRate(double permitsPerSecond) {
    checkArgument(permitsPerSecond > 0.0, "rate must be positive");
    synchronized (mutex()) {
      doSetRate(permitsPerSecond, stopwatch.readMicros());
    }
  }

  abstract void doSetRate(double permitsPerSecond, long nowMicros);

  /**
   * Returns the stable rate (as {@code permits per seconds}) with which this {@code RateLimiter} is
