// Source-based slice around line 852
// Method: <com.google.common.util.concurrent.Monitor: boolean waitFor(Guard,long,TimeUnit)>


  /**
   * Waits for the guard to be satisfied. Waits at most the given time, and may be interrupted. May
   * be called only by a thread currently occupying this monitor.
   *
   * @return whether the guard is now satisfied
   * @throws InterruptedException if interrupted while waiting
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean waitFor(Guard guard, long time, TimeUnit unit) throws InterruptedException {
    long timeoutNanos = toSafeNanos(time, unit);
    if (!((guard.monitor == this) && lock.isHeldByCurrentThread())) {
      throw new IllegalMonitorStateException();
    }
    if (guard.isSatisfied()) {
      return true;
    }
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
