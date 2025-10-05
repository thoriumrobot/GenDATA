// Source-based slice around line 1179
// Method: <com.google.common.util.concurrent.Monitor: void await(Guard,boolean)>

  }

  /*
   * Methods that loop waiting on a guard's condition until the guard is satisfied, while recording
   * this fact so that other threads know to check our guard and signal us. It's caller's
   * responsibility to ensure that the guard is *not* currently satisfied.
   */

  @GuardedBy("lock")
  private void await(Guard guard, boolean signalBeforeWaiting) throws InterruptedException {
    if (signalBeforeWaiting) {
      signalNextWaiter();
    }
    beginWaitingFor(guard);
    try {
      do {
        guard.condition.await();
      } while (!guard.isSatisfied());
    } finally {
      endWaitingFor(guard);
