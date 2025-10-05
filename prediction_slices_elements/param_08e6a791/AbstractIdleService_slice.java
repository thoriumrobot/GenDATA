// Source-based slice around line 210
// Method: <com.google.common.util.concurrent.AbstractIdleService: void awaitTerminated(long,TimeUnit)>

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

  /**
   * Returns the name of this service. {@link AbstractIdleService} may include the name in debugging
   * output.
   *
   * @since 14.0
   */
  protected String serviceName() {
