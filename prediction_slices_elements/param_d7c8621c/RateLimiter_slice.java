// Source-based slice around line 434
// Method: <com.google.common.util.concurrent.RateLimiter: long reserveAndGetWaitLength(int,long)>

  private boolean canAcquire(long nowMicros, long timeoutMicros) {
    return queryEarliestAvailable(nowMicros) - timeoutMicros <= nowMicros;
  }

  /**
   * Reserves next ticket and returns the wait time that the caller must wait for.
   *
   * @return the required wait time, never negative
   */
  final long reserveAndGetWaitLength(int permits, long nowMicros) {
    long momentAvailable = reserveEarliestAvailable(permits, nowMicros);
    return max(momentAvailable - nowMicros, 0);
  }

  /**
   * Returns the earliest time that permits are available (with one caveat).
   *
   * @return the time that permits are available, or, if permits are available immediately, an
   *     arbitrary past or present time
   */
