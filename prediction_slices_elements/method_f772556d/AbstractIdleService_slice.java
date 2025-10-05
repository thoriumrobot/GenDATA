// Source-based slice around line 161
// Method: <com.google.common.util.concurrent.AbstractIdleService: Service stopAsync()>

    delegate.startAsync();
    return this;
  }

  /**
   * @since 15.0
   */
  @CanIgnoreReturnValue
  @Override
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
