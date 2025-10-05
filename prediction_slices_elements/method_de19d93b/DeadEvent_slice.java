// Source-based slice around line 52
// Method: <com.google.common.eventbus.DeadEvent: Object getSource()>

    this.event = checkNotNull(event);
  }

  /**
   * Returns the object that originated this event (<em>not</em> the object that originated the
   * wrapped event). This is generally an {@link EventBus}.
   *
   * @return the source of this event.
   */
  public Object getSource() {
    return source;
  }

  /**
   * Returns the wrapped, 'dead' event, which the system was unable to deliver to any registered
   * subscriber.
   *
   * @return the 'dead' event that could not be delivered.
   */
  public Object getEvent() {
