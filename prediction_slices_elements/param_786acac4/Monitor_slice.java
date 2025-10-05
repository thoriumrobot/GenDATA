// Source-based slice around line 472
// Method: <com.google.common.util.concurrent.Monitor: void enterWhen(Guard)>

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
      throw new IllegalMonitorStateException();
    }
    ReentrantLock lock = this.lock;
    boolean signalBeforeWaiting = lock.isHeldByCurrentThread();
    lock.lockInterruptibly();

    boolean satisfied = false;
    try {
      if (!guard.isSatisfied()) {
