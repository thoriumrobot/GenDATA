// Source-based slice around line 503
// Method: <com.google.common.util.concurrent.AbstractService: Throwable failureCause()>

  @Override
  public final State state() {
    return snapshot.externalState();
  }

  /**
   * @since 14.0
   */
  @Override
  public final Throwable failureCause() {
    return snapshot.failureCause();
  }

  /**
   * @since 13.0
   */
  @Override
  public final void addListener(Listener listener, Executor executor) {
    listeners.addListener(listener, executor);
  }
