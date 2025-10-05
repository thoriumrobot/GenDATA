// Source-based slice around line 516
// Method: <com.google.common.util.concurrent.AbstractService: String toString()>

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
   * #monitor}.
   */
  private void dispatchListenerEvents() {
    if (!monitor.isOccupiedByCurrentThread()) {
      listeners.dispatch();
