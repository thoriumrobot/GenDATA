// Source-based slice around line 530
// Method: <com.google.common.util.concurrent.AbstractService: void enqueueStartingEvent()>

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
    listeners.enqueue(RUNNING_EVENT);
  }

  private void enqueueStoppingEvent(State from) {
    if (from == State.STARTING) {
      listeners.enqueue(STOPPING_FROM_STARTING_EVENT);
