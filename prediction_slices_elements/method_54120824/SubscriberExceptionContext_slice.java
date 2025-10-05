// Source-based slice around line 56
// Method: <com.google.common.eventbus.SubscriberExceptionContext: Object getEvent()>

  /**
   * Returns the {@link EventBus} that handled the event and the subscriber. Useful for broadcasting
   * a new event based on the error.
   */
  public EventBus getEventBus() {
    return eventBus;
  }

  /** Returns the event object that caused the subscriber to throw. */
  public Object getEvent() {
    return event;
  }

  /** Returns the object context that the subscriber was called on. */
  public Object getSubscriber() {
    return subscriber;
  }

  /** Returns the subscribed method that threw the exception. */
  public Method getSubscriberMethod() {
