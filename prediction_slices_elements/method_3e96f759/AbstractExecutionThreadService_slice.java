// Source-based slice around line 178
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: Throwable failureCause()>

  @Override
  public final void addListener(Listener listener, Executor executor) {
    delegate.addListener(listener, executor);
  }

  /**
   * @since 14.0
   */
  @Override
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
