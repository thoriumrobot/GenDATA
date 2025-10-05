// Source-based slice around line 151
// Method: <com.google.common.util.concurrent.AbstractIdleService: Service startAsync()>

  public final Throwable failureCause() {
    return delegate.failureCause();
  }

  /**
   * @since 15.0
   */
  @CanIgnoreReturnValue
  @Override
  public final Service startAsync() {
    delegate.startAsync();
    return this;
  }

  /**
   * @since 15.0
   */
  @CanIgnoreReturnValue
  @Override
  public final Service stopAsync() {
