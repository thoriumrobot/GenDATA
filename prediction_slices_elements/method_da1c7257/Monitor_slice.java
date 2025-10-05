// Source-based slice around line 1143
// Method: <com.google.common.util.concurrent.Monitor: void beginWaitingFor(Guard)>

  @GuardedBy("lock")
  private void signalAllWaiters() {
    for (Guard guard = activeGuards; guard != null; guard = guard.next) {
      guard.condition.signalAll();
    }
  }

  /** Records that the current thread is about to wait on the specified guard. */
  @GuardedBy("lock")
  private void beginWaitingFor(Guard guard) {
    int waiters = guard.waiterCount++;
    if (waiters == 0) {
      // push guard onto activeGuards
      guard.next = activeGuards;
      activeGuards = guard;
    }
  }

  /** Records that the current thread is no longer waiting on the specified guard. */
  @GuardedBy("lock")
