// Source-based slice around line 61
// Method: <com.google.common.eventbus.SubscriberExceptionContext: Object getSubscriber()>

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
    return subscriberMethod;
  }
}
