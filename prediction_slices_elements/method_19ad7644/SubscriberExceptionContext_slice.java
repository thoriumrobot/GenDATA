// Source-based slice around line 66
// Method: <com.google.common.eventbus.SubscriberExceptionContext: Method getSubscriberMethod()>

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
