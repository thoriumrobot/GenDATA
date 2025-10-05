// Source-based slice around line 246
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: void awaitTerminated(long,TimeUnit)>

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
   * Returns the name of this service. {@link AbstractExecutionThreadService} may include the name
   * in debugging output.
   *
   * <p>Subclasses may override this method.
   *
   * @since 14.0 (present in 10.0 as getServiceName)
