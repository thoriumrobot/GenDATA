// Source-based slice around line 315
// Method: <com.google.common.util.concurrent.RateLimiter: long reserve(int)>

    return 1.0 * microsToWait / SECONDS.toMicros(1L);
  }

  /**
   * Reserves the given number of permits from this {@code RateLimiter} for future use, returning
   * the number of microseconds until the reservation can be consumed.
   *
   * @return time in microseconds to wait until the resource can be acquired, never negative
   */
  final long reserve(int permits) {
    checkPermits(permits);
    synchronized (mutex()) {
      return reserveAndGetWaitLength(permits, stopwatch.readMicros());
    }
  }

  /**
   * Acquires a permit from this {@code RateLimiter} if it can be obtained without exceeding the
   * specified {@code timeout}, or returns {@code false} immediately (without waiting) if the permit
   * would not have been granted before the timeout expired.
