// Source-based slice around line 870
// Method: <com.google.common.util.concurrent.Monitor: void waitForUninterruptibly(Guard)>

      throw new InterruptedException();
    }
    return awaitNanos(guard, timeoutNanos, true);
  }

  /**
   * Waits for the guard to be satisfied. Waits indefinitely. May be called only by a thread
   * currently occupying this monitor.
   */
  public void waitForUninterruptibly(Guard guard) {
    if (!((guard.monitor == this) && lock.isHeldByCurrentThread())) {
      throw new IllegalMonitorStateException();
    }
    if (!guard.isSatisfied()) {
      awaitUninterruptibly(guard, true);
    }
  }

  /**
   * Waits for the guard to be satisfied. Waits at most the given time. May be called only by a
