// Source-based slice around line 548
// Method: <com.google.common.util.concurrent.AbstractService: void enqueueTerminatedEvent(State)>

    if (from == State.STARTING) {
      listeners.enqueue(STOPPING_FROM_STARTING_EVENT);
    } else if (from == State.RUNNING) {
      listeners.enqueue(STOPPING_FROM_RUNNING_EVENT);
    } else {
      throw new AssertionError();
    }
  }

  private void enqueueTerminatedEvent(State from) {
    switch (from) {
      case NEW:
        listeners.enqueue(TERMINATED_FROM_NEW_EVENT);
        break;
      case STARTING:
        listeners.enqueue(TERMINATED_FROM_STARTING_EVENT);
        break;
      case RUNNING:
        listeners.enqueue(TERMINATED_FROM_RUNNING_EVENT);
        break;
