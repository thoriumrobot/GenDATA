// Source-based slice around line 457
// Method: <com.google.common.util.concurrent.RateLimiter: String toString()>

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

    /*
     * We always hold the mutex when calling this. TODO(cpovirk): Is that important? Perhaps we need
     * to guarantee that each call to reserveEarliestAvailable, etc. sees a value >= the previous?
