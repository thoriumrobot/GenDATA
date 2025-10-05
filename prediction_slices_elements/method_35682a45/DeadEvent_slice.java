// Source-based slice around line 67
// Method: <com.google.common.eventbus.DeadEvent: String toString()>

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
