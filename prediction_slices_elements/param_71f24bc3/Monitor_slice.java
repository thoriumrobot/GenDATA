// Source-based slice around line 1210
// Method: <com.google.common.util.concurrent.Monitor: boolean awaitNanos(Guard,long,boolean)>

        guard.condition.awaitUninterruptibly();
      } while (!guard.isSatisfied());
    } finally {
      endWaitingFor(guard);
    }
  }

  /** Caller should check before calling that guard is not satisfied. */
  @GuardedBy("lock")
  private boolean awaitNanos(Guard guard, long nanos, boolean signalBeforeWaiting)
      throws InterruptedException {
    boolean firstTime = true;
    try {
      do {
        if (nanos <= 0L) {
          return false;
        }
        if (firstTime) {
          if (signalBeforeWaiting) {
            signalNextWaiter();
