// Source-based slice around line 929
// Method: <com.google.common.util.concurrent.Monitor: void leave()>

      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }

  /** Leaves this monitor. May be called only by a thread currently occupying this monitor. */
  public void leave() {
    ReentrantLock lock = this.lock;
    try {
      // No need to signal if we will still be holding the lock when we return
      if (lock.getHoldCount() == 1) {
        signalNextWaiter();
      }
    } finally {
      lock.unlock(); // Will throw IllegalMonitorStateException if not held
    }
  }
