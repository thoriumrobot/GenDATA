// Source-based slice around line 186
// Method: <com.google.common.util.concurrent.AbstractIdleService: void awaitRunning(long,TimeUnit)>

  @Override
  public final void awaitRunning(Duration timeout) throws TimeoutException {
    Service.super.awaitRunning(timeout);
  }

  /**
   * @since 15.0
   */
  @Override
  public final void awaitRunning(long timeout, TimeUnit unit) throws TimeoutException {
    delegate.awaitRunning(timeout, unit);
  }

  /**
   * @since 15.0
   */
  @Override
  public final void awaitTerminated() {
    delegate.awaitTerminated();
  }
