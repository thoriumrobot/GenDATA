// Source-based slice around line 126
// Method: <com.google.common.util.concurrent.AbstractIdleService: State state()>

    return serviceName() + " [" + state() + "]";
  }

  @Override
  public final boolean isRunning() {
    return delegate.isRunning();
  }

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
