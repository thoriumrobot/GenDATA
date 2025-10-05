// Source-based slice around line 518
// Method: <com.google.common.util.concurrent.Monitor: boolean enterWhen(Guard,long,TimeUnit)>

   * interrupted.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   * @throws InterruptedException if interrupted while waiting
   */
  @SuppressWarnings({
    "GoodTime", // should accept a java.time.Duration
    "LabelledBreakTarget", // TODO(b/345814817): Maybe fix.
  })
  public boolean enterWhen(Guard guard, long time, TimeUnit unit) throws InterruptedException {
    long timeoutNanos = toSafeNanos(time, unit);
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    boolean reentrant = lock.isHeldByCurrentThread();
    long startTime = 0L;

    locked:
    {
