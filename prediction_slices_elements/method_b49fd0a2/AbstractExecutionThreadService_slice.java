// Source-based slice around line 206
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: void awaitRunning()>

  public final Service stopAsync() {
    delegate.stopAsync();
    return this;
  }

  /**
   * @since 15.0
   */
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
