// Source-based slice around line 823
// Method: <com.google.common.util.concurrent.Monitor: void waitFor(Guard)>

    }
  }

  /**
   * Waits for the guard to be satisfied. Waits indefinitely, but may be interrupted. May be called
   * only by a thread currently occupying this monitor.
   *
   * @throws InterruptedException if interrupted while waiting
   */
  public void waitFor(Guard guard) throws InterruptedException {
    if (!((guard.monitor == this) && lock.isHeldByCurrentThread())) {
      throw new IllegalMonitorStateException();
    }
    if (!guard.isSatisfied()) {
      await(guard, true);
    }
  }

  /**
   * Waits for the guard to be satisfied. Waits at most the given time, and may be interrupted. May
