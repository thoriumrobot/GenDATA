// Source-based slice around line 194
// Method: <com.google.common.util.concurrent.AbstractIdleService: void awaitTerminated()>

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

  /**
   * @since 28.0
   */
  @Override
  public final void awaitTerminated(Duration timeout) throws TimeoutException {
    Service.super.awaitTerminated(timeout);
  }
