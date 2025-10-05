// Source-based slice around line 352
// Method: <com.google.common.util.concurrent.AbstractService: void awaitTerminated(Duration)>

    } finally {
      monitor.leave();
    }
  }

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
