// Source-based slice around line 1123
// Method: <com.google.common.util.concurrent.Monitor: boolean isSatisfied(Guard)>

  //       }
  //     }
  //   }

  /**
   * Exactly like guard.isSatisfied(), but in addition signals all waiting threads in the (hopefully
   * unlikely) event that isSatisfied() throws.
   */
  @GuardedBy("lock")
  private boolean isSatisfied(Guard guard) {
    try {
      return guard.isSatisfied();
    } catch (Throwable throwable) {
      // Any Exception is either a RuntimeException or sneaky checked exception.
      signalAllWaiters();
      throw throwable;
    }
  }

  /** Signals all threads waiting on guards. */
