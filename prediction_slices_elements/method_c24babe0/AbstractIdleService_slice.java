// Source-based slice around line 178
// Method: <com.google.common.util.concurrent.AbstractIdleService: void awaitRunning(Duration)>

  @Override
  public final void awaitRunning() {
    delegate.awaitRunning();
  }

  /**
   * @since 28.0
   */
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
