// Source-based slice around line 1042
// Method: <com.google.common.util.concurrent.Monitor: long initNanoTime(long)>

  private static long toSafeNanos(long time, TimeUnit unit) {
    long timeoutNanos = unit.toNanos(time);
    return Longs.constrainToRange(timeoutNanos, 0L, (Long.MAX_VALUE / 4) * 3);
  }

  /**
   * Returns System.nanoTime() unless the timeout has already elapsed. Returns 0L if and only if the
   * timeout has already elapsed.
   */
  private static long initNanoTime(long timeoutNanos) {
    if (timeoutNanos <= 0L) {
      return 0L;
    } else {
      long startTime = System.nanoTime();
      return (startTime == 0L) ? 1L : startTime;
    }
  }

  /**
   * Returns the remaining nanos until the given timeout, or 0L if the timeout has already elapsed.
