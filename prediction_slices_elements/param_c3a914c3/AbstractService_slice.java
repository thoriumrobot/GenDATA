// Source-based slice around line 511
// Method: <com.google.common.util.concurrent.AbstractService: void addListener(Listener,Executor)>

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

  @Override
  public String toString() {
    return getClass().getSimpleName() + " [" + state() + "]";
  }

  /**
   * Attempts to execute all the listeners in {@link #listeners} while not holding the {@link
