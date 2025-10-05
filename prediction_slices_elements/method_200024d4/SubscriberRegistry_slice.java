// Source-based slice around line 117
// Method: <com.google.common.eventbus.SubscriberRegistry: Set getSubscribersForTesting(Class)>

            "missing event subscriber for an annotated method. Is " + listener + " registered?");
      }

      // don't try to remove the set if it's empty; that can't be done safely without a lock
      // anyway, if the set is empty it'll just be wrapping an array of length 0
    }
  }

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

