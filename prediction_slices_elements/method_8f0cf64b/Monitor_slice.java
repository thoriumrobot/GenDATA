// Source-based slice around line 1091
// Method: <com.google.common.util.concurrent.Monitor: void signalNextWaiter()>

   * happen due to spurious wakeup (ignorable) or another thread acquiring the lock before the
   * current thread can and returning the guard to the unsatisfied state. In the latter case the
   * other thread (last thread modifying the state protected by the monitor) takes over the
   * responsibility of signalling the next waiter.
   *
   * <p>This method must not be called from within a beginWaitingFor/endWaitingFor block, or else
   * the current thread's guard might be mistakenly signalled, leading to a lost signal.
   */
  @GuardedBy("lock")
  private void signalNextWaiter() {
    for (Guard guard = activeGuards; guard != null; guard = guard.next) {
      if (isSatisfied(guard)) {
        guard.condition.signal();
        break;
      }
    }
  }

  /**
   * Exactly like signalNextWaiter, but caller guarantees that guardToSkip need not be considered,
