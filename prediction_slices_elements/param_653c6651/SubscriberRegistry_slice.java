// Source-based slice around line 125
// Method: <com.google.common.eventbus.SubscriberRegistry: Iterator getSubscribers(Object)>

  @VisibleForTesting
  Set<Subscriber> getSubscribersForTesting(Class<?> eventType) {
    return MoreObjects.firstNonNull(subscribers.get(eventType), ImmutableSet.<Subscriber>of());
  }

  /**
   * Gets an iterator representing an immutable snapshot of all subscribers to the given event at
   * the time this method is called.
   */
  Iterator<Subscriber> getSubscribers(Object event) {
    ImmutableSet<Class<?>> eventTypes = flattenHierarchy(event.getClass());

    List<Iterator<Subscriber>> subscriberIterators =
        Lists.newArrayListWithCapacity(eventTypes.size());

    for (Class<?> eventType : eventTypes) {
      CopyOnWriteArraySet<Subscriber> eventSubscribers = subscribers.get(eventType);
      if (eventSubscribers != null) {
        // eager no-copy snapshot
        subscriberIterators.add(eventSubscribers.iterator());
