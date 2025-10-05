// Source-based slice around line 1154
// Method: <com.google.common.util.concurrent.Monitor: void endWaitingFor(Guard)>

    if (waiters == 0) {
      // push guard onto activeGuards
      guard.next = activeGuards;
      activeGuards = guard;
    }
  }

  /** Records that the current thread is no longer waiting on the specified guard. */
  @GuardedBy("lock")
  private void endWaitingFor(Guard guard) {
    int waiters = --guard.waiterCount;
    if (waiters == 0) {
      // unlink guard from activeGuards
      for (Guard p = activeGuards, pred = null; ; pred = p, p = p.next) {
        if (p == guard) {
          if (pred == null) {
            activeGuards = p.next;
          } else {
            pred.next = p.next;
          }
