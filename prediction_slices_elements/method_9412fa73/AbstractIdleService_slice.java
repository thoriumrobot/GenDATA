// Source-based slice around line 121
// Method: <com.google.common.util.concurrent.AbstractIdleService: boolean isRunning()>

    return command -> newThread(threadNameSupplier.get(), command).start();
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
