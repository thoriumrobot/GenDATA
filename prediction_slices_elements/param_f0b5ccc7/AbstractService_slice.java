// Source-based slice around line 568
// Method: <com.google.common.util.concurrent.AbstractService: void enqueueFailedEvent(State,Throwable)>

      case STOPPING:
        listeners.enqueue(TERMINATED_FROM_STOPPING_EVENT);
        break;
      case TERMINATED:
      case FAILED:
        throw new AssertionError();
    }
  }

  private void enqueueFailedEvent(State from, Throwable cause) {
    // can't memoize this one due to the exception
    listeners.enqueue(
        new ListenerCallQueue.Event<Listener>() {
          @Override
          public void call(Listener listener) {
            listener.failed(from, cause);
          }

          @Override
          public String toString() {
