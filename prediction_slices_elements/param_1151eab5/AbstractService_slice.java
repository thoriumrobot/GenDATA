// Source-based slice around line 94
// Method: <com.google.common.util.concurrent.AbstractService: ListenerCallQueue terminatedEvent(State)>

  private static final ListenerCallQueue.Event<Listener> TERMINATED_FROM_NEW_EVENT =
      terminatedEvent(NEW);
  private static final ListenerCallQueue.Event<Listener> TERMINATED_FROM_STARTING_EVENT =
      terminatedEvent(STARTING);
  private static final ListenerCallQueue.Event<Listener> TERMINATED_FROM_RUNNING_EVENT =
      terminatedEvent(RUNNING);
  private static final ListenerCallQueue.Event<Listener> TERMINATED_FROM_STOPPING_EVENT =
      terminatedEvent(STOPPING);

  private static ListenerCallQueue.Event<Listener> terminatedEvent(State from) {
    return new ListenerCallQueue.Event<Listener>() {
      @Override
      public void call(Listener listener) {
        listener.terminated(from);
      }

      @Override
      public String toString() {
        return "terminated({from = " + from + "})";
      }
