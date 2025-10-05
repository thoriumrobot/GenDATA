// Source-based slice around line 202
// Method: <com.google.common.util.concurrent.AbstractIdleService: void awaitTerminated(Duration)>

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

  /**
   * @since 15.0
   */
  @Override
  public final void awaitTerminated(long timeout, TimeUnit unit) throws TimeoutException {
    delegate.awaitTerminated(timeout, unit);
  }
