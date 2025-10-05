// Source-based slice around line 94
// Method: <com.google.common.eventbus.SubscriberRegistry: void unregister(Object)>

        eventSubscribers =
            MoreObjects.firstNonNull(subscribers.putIfAbsent(eventType, newSet), newSet);
      }

      eventSubscribers.addAll(eventMethodsInListener);
    }
  }

  /** Unregisters all subscribers on the given listener object. */
  void unregister(Object listener) {
    Multimap<Class<?>, Subscriber> listenerMethods = findAllSubscribers(listener);

    for (Entry<Class<?>, Collection<Subscriber>> entry : listenerMethods.asMap().entrySet()) {
      Class<?> eventType = entry.getKey();
      Collection<Subscriber> listenerMethodsForType = entry.getValue();

      CopyOnWriteArraySet<Subscriber> currentSubscribers = subscribers.get(eventType);
      if (currentSubscribers == null || !currentSubscribers.removeAll(listenerMethodsForType)) {
        // if removeAll returns true, all we really know is that at least one subscriber was
        // removed... however, barring something very strange we can assume that if at least one
