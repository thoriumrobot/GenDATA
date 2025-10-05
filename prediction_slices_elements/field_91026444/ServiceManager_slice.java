// Source-based slice around line 139
// Method: com.google.common.util.concurrent.ServiceManager.STOPPED_EVENT

        public void call(Listener listener) {
          listener.healthy();
        }

        @Override
        public String toString() {
          return "healthy()";
        }
      };
  private static final ListenerCallQueue.Event<Listener> STOPPED_EVENT =
      new ListenerCallQueue.Event<Listener>() {
        @Override
        public void call(Listener listener) {
          listener.stopped();
        }

        @Override
        public String toString() {
          return "stopped()";
        }
