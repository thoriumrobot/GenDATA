// Source-based slice around line 322
// Method: <com.google.common.util.concurrent.AbstractService: void awaitRunning(long,TimeUnit)>

  /**
   * @since 28.0
   */
  @Override
  public final void awaitRunning(Duration timeout) throws TimeoutException {
    Service.super.awaitRunning(timeout);
  }

  @Override
  public final void awaitRunning(long timeout, TimeUnit unit) throws TimeoutException {
    if (monitor.enterWhenUninterruptibly(hasReachedRunning, timeout, unit)) {
      try {
        checkCurrentState(RUNNING);
      } finally {
        monitor.leave();
      }
    } else {
      // It is possible due to races that we are currently in the expected state even though we
      // timed out. e.g. if we weren't event able to grab the lock within the timeout we would never
      // even check the guard. I don't think we care too much about this use case but it could lead
