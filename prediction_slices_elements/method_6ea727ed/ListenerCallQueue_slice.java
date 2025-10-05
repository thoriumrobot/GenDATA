// Source-based slice around line 118
// Method: <com.google.common.util.concurrent.ListenerCallQueue: void dispatch()>

      }
    }
  }

  /**
   * Dispatches all events enqueued prior to this call, serially and in order, for every listener.
   *
   * <p>Note: this method is idempotent and safe to call from any thread
   */
  public void dispatch() {
    // iterate by index to avoid concurrent modification exceptions
    for (int i = 0; i < listeners.size(); i++) {
      listeners.get(i).dispatch();
    }
  }

  /**
   * A special purpose queue/executor that dispatches listener events serially on a configured
   * executor. Each event can be added and dispatched as separate phases.
   *
