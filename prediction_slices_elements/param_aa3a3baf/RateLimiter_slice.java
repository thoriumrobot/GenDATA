// Source-based slice around line 445
// Method: <com.google.common.util.concurrent.RateLimiter: long queryEarliestAvailable(long)>

    return max(momentAvailable - nowMicros, 0);
  }

  /**
   * Returns the earliest time that permits are available (with one caveat).
   *
   * @return the time that permits are available, or, if permits are available immediately, an
   *     arbitrary past or present time
   */
  abstract long queryEarliestAvailable(long nowMicros);

  /**
   * Reserves the requested number of permits and returns the time that those permits can be used
   * (with one caveat).
   *
   * @return the time that the permits may be used, or, if the permits may be used immediately, an
   *     arbitrary past or present time
   */
  abstract long reserveEarliestAvailable(int permits, long nowMicros);

