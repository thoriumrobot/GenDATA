// Source-based slice around line 89
// Method: <com.google.common.util.concurrent.ListenerCallQueue: void enqueue(Event)>


  /**
   * Enqueues an event to be run on currently known listeners.
   *
   * <p>The {@code toString} method of the Event itself will be used to describe the event in the
   * case of an error.
   *
   * @param event the callback to execute on {@link #dispatch}
   */
  public void enqueue(Event<L> event) {
    enqueueHelper(event, event);
  }

  /**
   * Enqueues an event to be run on currently known listeners, with a label.
   *
   * @param event the callback to execute on {@link #dispatch}
   * @param label a description of the event to use in the case of an error
   */
  public void enqueue(Event<L> event, String label) {
