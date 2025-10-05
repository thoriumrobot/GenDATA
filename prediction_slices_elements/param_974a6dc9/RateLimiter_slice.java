// Source-based slice around line 425
// Method: <com.google.common.util.concurrent.RateLimiter: boolean canAcquire(long,long)>

        return false;
      } else {
        microsToWait = reserveAndGetWaitLength(permits, nowMicros);
      }
    }
    stopwatch.sleepMicrosUninterruptibly(microsToWait);
    return true;
  }

  private boolean canAcquire(long nowMicros, long timeoutMicros) {
    return queryEarliestAvailable(nowMicros) - timeoutMicros <= nowMicros;
  }

  /**
   * Reserves next ticket and returns the wait time that the caller must wait for.
   *
   * @return the required wait time, never negative
   */
  final long reserveAndGetWaitLength(int permits, long nowMicros) {
    long momentAvailable = reserveEarliestAvailable(permits, nowMicros);
