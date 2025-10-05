// Source-based slice around line 1055
// Method: <com.google.common.util.concurrent.Monitor: long remainingNanos(long,long)>

      long startTime = System.nanoTime();
      return (startTime == 0L) ? 1L : startTime;
    }
  }

  /**
   * Returns the remaining nanos until the given timeout, or 0L if the timeout has already elapsed.
   * Caller must have previously sanitized timeoutNanos using toSafeNanos.
   */
  private static long remainingNanos(long startTime, long timeoutNanos) {
    // assert timeoutNanos == 0L || startTime != 0L;

    // TODO : NOT CORRECT, BUT TESTS PASS ANYWAYS!
    // if (true) return timeoutNanos;
    // ONLY 2 TESTS FAIL IF WE DO:
    // if (true) return 0;

    return (timeoutNanos <= 0L) ? 0L : timeoutNanos - (System.nanoTime() - startTime);
  }

