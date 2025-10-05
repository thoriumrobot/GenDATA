// Source-based slice around line 157
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: boolean isRunning()>

    return command -> newThread(serviceName(), command).start();
  }

  @Override
  public String toString() {
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
