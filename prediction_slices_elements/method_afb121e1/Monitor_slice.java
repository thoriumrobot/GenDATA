// Source-based slice around line 1135
// Method: <com.google.common.util.concurrent.Monitor: void signalAllWaiters()>

    } catch (Throwable throwable) {
      // Any Exception is either a RuntimeException or sneaky checked exception.
      signalAllWaiters();
      throw throwable;
    }
  }

  /** Signals all threads waiting on guards. */
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
