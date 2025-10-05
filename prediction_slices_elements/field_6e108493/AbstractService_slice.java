// Source-based slice around line 68
// Method: com.google.common.util.concurrent.AbstractService.RUNNING_EVENT

        public void call(Listener listener) {
          listener.starting();
        }

        @Override
        public String toString() {
          return "starting()";
        }
      };
  private static final ListenerCallQueue.Event<Listener> RUNNING_EVENT =
      new ListenerCallQueue.Event<Listener>() {
        @Override
        public void call(Listener listener) {
          listener.running();
        }

        @Override
        public String toString() {
          return "running()";
        }
