// Source-based slice around line 897
// Method: <com.google.common.util.concurrent.Monitor: boolean waitForUninterruptibly(Guard,long,TimeUnit)>

  }

  /**
   * Waits for the guard to be satisfied. Waits at most the given time. May be called only by a
   * thread currently occupying this monitor.
   *
   * @return whether the guard is now satisfied
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean waitForUninterruptibly(Guard guard, long time, TimeUnit unit) {
    long timeoutNanos = toSafeNanos(time, unit);
    if (!((guard.monitor == this) && lock.isHeldByCurrentThread())) {
      throw new IllegalMonitorStateException();
    }
    if (guard.isSatisfied()) {
      return true;
    }
    boolean signalBeforeWaiting = true;
    long startTime = initNanoTime(timeoutNanos);
    boolean interrupted = Thread.interrupted();
