// Source-based slice around line 62
// Method: <com.google.common.eventbus.DeadEvent: Object getEvent()>

    return source;
  }

  /**
   * Returns the wrapped, 'dead' event, which the system was unable to deliver to any registered
   * subscriber.
   *
   * @return the 'dead' event that could not be delivered.
   */
  public Object getEvent() {
    return event;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("source", source).add("event", event).toString();
  }
}
