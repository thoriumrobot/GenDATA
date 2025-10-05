// Source-based slice around line 108
// Method: <com.google.common.util.concurrent.AbstractService: ListenerCallQueue stoppingEvent(State)>

      }

      @Override
      public String toString() {
        return "terminated({from = " + from + "})";
      }
    };
  }

  private static ListenerCallQueue.Event<Listener> stoppingEvent(State from) {
    return new ListenerCallQueue.Event<Listener>() {
      @Override
      public void call(Listener listener) {
        listener.stopping(from);
      }

      @Override
      public String toString() {
        return "stopping({from = " + from + "})";
      }
