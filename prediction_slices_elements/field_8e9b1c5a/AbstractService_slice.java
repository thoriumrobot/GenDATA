// Source-based slice around line 87
// Method: com.google.common.util.concurrent.AbstractService.TERMINATED_FROM_STARTING_EVENT

        }
      };
  private static final ListenerCallQueue.Event<Listener> STOPPING_FROM_STARTING_EVENT =
      stoppingEvent(STARTING);
  private static final ListenerCallQueue.Event<Listener> STOPPING_FROM_RUNNING_EVENT =
      stoppingEvent(RUNNING);

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
