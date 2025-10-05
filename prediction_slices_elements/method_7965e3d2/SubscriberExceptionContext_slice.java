// Source-based slice around line 51
// Method: <com.google.common.eventbus.SubscriberExceptionContext: EventBus getEventBus()>

    this.event = checkNotNull(event);
    this.subscriber = checkNotNull(subscriber);
    this.subscriberMethod = checkNotNull(subscriberMethod);
  }

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
