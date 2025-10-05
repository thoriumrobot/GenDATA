// Source-based slice around line 463
// Method: <com.google.common.util.concurrent.Monitor: boolean tryEnter()>

  }

  /**
   * Enters this monitor if it is possible to do so immediately. Does not block.
   *
   * <p><b>Note:</b> This method disregards the fairness setting of this monitor.
   *
   * @return whether the monitor was entered
   */
  public boolean tryEnter() {
    return lock.tryLock();
  }

  /**
   * Enters this monitor when the guard is satisfied. Blocks indefinitely, but may be interrupted.
   *
   * @throws InterruptedException if interrupted while waiting
   */
  public void enterWhen(Guard guard) throws InterruptedException {
    if (guard.monitor != this) {
