// Source-based slice around line 524
// Method: <com.google.common.util.concurrent.AbstractService: void dispatchListenerEvents()>

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
    }
  }

  private void enqueueStartingEvent() {
    listeners.enqueue(STARTING_EVENT);
  }

  private void enqueueRunningEvent() {
