// Source-based slice around line 454
// Method: <com.google.common.util.concurrent.RateLimiter: long reserveEarliestAvailable(int,long)>

  abstract long queryEarliestAvailable(long nowMicros);

  /**
   * Reserves the requested number of permits and returns the time that those permits can be used
   * (with one caveat).
   *
   * @return the time that the permits may be used, or, if the permits may be used immediately, an
   *     arbitrary past or present time
   */
  abstract long reserveEarliestAvailable(int permits, long nowMicros);

  @Override
  public String toString() {
    return String.format(Locale.ROOT, "RateLimiter[stableRate=%3.1fqps]", getRate());
  }

  abstract static class SleepingStopwatch {
    /** Constructor for use by subclasses. */
    protected SleepingStopwatch() {}

