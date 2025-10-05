// Source-based slice around line 134
// Method: <com.google.common.util.concurrent.AbstractIdleService: void addListener(Listener,Executor)>

  @Override
  public final State state() {
    return delegate.state();
  }

  /**
   * @since 13.0
   */
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
