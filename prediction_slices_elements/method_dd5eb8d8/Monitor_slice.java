// Source-based slice around line 570
// Method: <com.google.common.util.concurrent.Monitor: void enterWhenUninterruptibly(Guard)>

          }
        } finally {
          lock.unlock();
        }
      }
    }
  }

  /** Enters this monitor when the guard is satisfied. Blocks indefinitely. */
  public void enterWhenUninterruptibly(Guard guard) {
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    boolean signalBeforeWaiting = lock.isHeldByCurrentThread();
    lock.lock();

    boolean satisfied = false;
    try {
      if (!guard.isSatisfied()) {
