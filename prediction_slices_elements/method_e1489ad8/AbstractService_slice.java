// Source-based slice around line 357
// Method: <com.google.common.util.concurrent.AbstractService: void awaitTerminated(long,TimeUnit)>

  /**
   * @since 28.0
   */
  @Override
  public final void awaitTerminated(Duration timeout) throws TimeoutException {
    Service.super.awaitTerminated(timeout);
  }

  @Override
  public final void awaitTerminated(long timeout, TimeUnit unit) throws TimeoutException {
    if (monitor.enterWhenUninterruptibly(isStopped, timeout, unit)) {
      try {
        checkCurrentState(TERMINATED);
      } finally {
        monitor.leave();
      }
    } else {
      // It is possible due to races that we are currently in the expected state even though we
      // timed out. e.g. if we weren't event able to grab the lock within the timeout we would never
      // even check the guard. I don't think we care too much about this use case but it could lead
