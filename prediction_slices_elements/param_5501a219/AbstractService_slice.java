// Source-based slice around line 317
// Method: <com.google.common.util.concurrent.AbstractService: void awaitRunning(Duration)>

    } finally {
      monitor.leave();
    }
  }

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
