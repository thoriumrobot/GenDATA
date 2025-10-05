// Source-based slice around line 430
// Method: <com.google.common.util.concurrent.Monitor: void enterInterruptibly()>

      }
    }
  }

  /**
   * Enters this monitor. Blocks indefinitely, but may be interrupted.
   *
   * @throws InterruptedException if interrupted while waiting
   */
  public void enterInterruptibly() throws InterruptedException {
    lock.lockInterruptibly();
  }

  /**
   * Enters this monitor. Blocks at most the given time, and may be interrupted.
   *
   * @return whether the monitor was entered
   * @throws InterruptedException if interrupted while waiting
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
