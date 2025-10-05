// Source-based slice around line 80
// Method: com.google.common.util.concurrent.AbstractService.STOPPING_FROM_STARTING_EVENT

        public void call(Listener listener) {
          listener.running();
        }

        @Override
        public String toString() {
          return "running()";
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
