// Source-based slice around line 942
// Method: <com.google.common.util.concurrent.Monitor: boolean isFair()>

      if (lock.getHoldCount() == 1) {
        signalNextWaiter();
      }
    } finally {
      lock.unlock(); // Will throw IllegalMonitorStateException if not held
    }
  }

  /** Returns whether this monitor is using a fair ordering policy. */
  public boolean isFair() {
    return fair;
  }

  /**
   * Returns whether this monitor is occupied by any thread. This method is designed for use in
   * monitoring of the system state, not for synchronization control.
   */
  public boolean isOccupied() {
    return lock.isLocked();
  }
