// Source-based slice around line 1194
// Method: <com.google.common.util.concurrent.Monitor: void awaitUninterruptibly(Guard,boolean)>

      do {
        guard.condition.await();
      } while (!guard.isSatisfied());
    } finally {
      endWaitingFor(guard);
    }
  }

  @GuardedBy("lock")
  private void awaitUninterruptibly(Guard guard, boolean signalBeforeWaiting) {
    if (signalBeforeWaiting) {
      signalNextWaiter();
    }
    beginWaitingFor(guard);
    try {
      do {
        guard.condition.awaitUninterruptibly();
      } while (!guard.isSatisfied());
    } finally {
      endWaitingFor(guard);
