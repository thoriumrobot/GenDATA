// Source-based slice around line 103
// Method: <com.google.common.util.concurrent.ListenerCallQueue: void enqueueHelper(Event,Object)>

   * Enqueues an event to be run on currently known listeners, with a label.
   *
   * @param event the callback to execute on {@link #dispatch}
   * @param label a description of the event to use in the case of an error
   */
  public void enqueue(Event<L> event, String label) {
    enqueueHelper(event, label);
  }

  private void enqueueHelper(Event<L> event, Object label) {
    checkNotNull(event, "event");
    checkNotNull(label, "label");
    synchronized (listeners) {
      for (PerListenerQueue<L> queue : listeners) {
        queue.add(event, label);
      }
    }
  }

  /**
