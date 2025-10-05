// Source-based slice around line 1033
// Method: <com.google.common.util.concurrent.Monitor: long toSafeNanos(long,TimeUnit)>

      lock.unlock();
    }
  }

  /**
   * Returns unit.toNanos(time), additionally ensuring the returned value is not at risk of
   * overflowing or underflowing, by bounding the value between 0 and (Long.MAX_VALUE / 4) * 3.
   * Actually waiting for more than 219 years is not supported!
   */
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
