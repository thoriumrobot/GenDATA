// Source-based slice around line 409
// Method: <com.google.common.util.concurrent.RateLimiter: boolean tryAcquire(int,long,TimeUnit)>

   * waiting) if the permits would not have been granted before the timeout expired.
   *
   * @param permits the number of permits to acquire
   * @param timeout the maximum time to wait for the permits. Negative values are treated as zero.
   * @param unit the time unit of the timeout argument
   * @return {@code true} if the permits were acquired, {@code false} otherwise
   * @throws IllegalArgumentException if the requested number of permits is negative or zero
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean tryAcquire(int permits, long timeout, TimeUnit unit) {
    long timeoutMicros = max(unit.toMicros(timeout), 0);
    checkPermits(permits);
    long microsToWait;
    synchronized (mutex()) {
      long nowMicros = stopwatch.readMicros();
      if (!canAcquire(nowMicros, timeoutMicros)) {
        return false;
      } else {
        microsToWait = reserveAndGetWaitLength(permits, nowMicros);
      }
