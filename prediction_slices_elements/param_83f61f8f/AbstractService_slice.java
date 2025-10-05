// Source-based slice around line 538
// Method: <com.google.common.util.concurrent.AbstractService: void enqueueStoppingEvent(State)>


  private void enqueueStartingEvent() {
    listeners.enqueue(STARTING_EVENT);
  }

  private void enqueueRunningEvent() {
    listeners.enqueue(RUNNING_EVENT);
  }

  private void enqueueStoppingEvent(State from) {
    if (from == State.STARTING) {
      listeners.enqueue(STOPPING_FROM_STARTING_EVENT);
    } else if (from == State.RUNNING) {
      listeners.enqueue(STOPPING_FROM_RUNNING_EVENT);
    } else {
      throw new AssertionError();
    }
  }

  private void enqueueTerminatedEvent(State from) {
