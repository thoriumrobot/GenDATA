// Source-based slice around line 75
// Method: <com.google.common.util.concurrent.ListenerCallQueue: void addListener(L,Executor)>

  interface Event<L> {
    /** Call a method on the listener. */
    void call(L listener);
  }

  /**
   * Adds a listener that will be called using the given executor when events are later {@link
   * #enqueue enqueued} and {@link #dispatch dispatched}.
   */
  public void addListener(L listener, Executor executor) {
    checkNotNull(listener, "listener");
    checkNotNull(executor, "executor");
    listeners.add(new PerListenerQueue<>(listener, executor));
  }

  /**
   * Enqueues an event to be run on currently known listeners.
   *
   * <p>The {@code toString} method of the Event itself will be used to describe the event in the
   * case of an error.
