// Source-based slice around line 710
// Method: <com.google.common.util.concurrent.Monitor: boolean enterIf(Guard,long,TimeUnit)>

  }

  /**
   * Enters this monitor if the guard is satisfied. Blocks at most the given time acquiring the
   * lock, but does not wait for the guard to be satisfied.
   *
   * @return whether the monitor was entered, which guarantees that the guard is now satisfied
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean enterIf(Guard guard, long time, TimeUnit unit) {
    if (guard.monitor != this) {
      throw new IllegalMonitorStateException();
    }
    if (!enter(time, unit)) {
      return false;
    }

    boolean satisfied = false;
    try {
      return satisfied = guard.isSatisfied();
