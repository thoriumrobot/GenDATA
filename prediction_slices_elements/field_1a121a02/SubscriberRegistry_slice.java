// Source-based slice around line 148
// Method: com.google.common.eventbus.SubscriberRegistry.subscriberMethodsCache

    return Iterators.concat(subscriberIterators.iterator());
  }

  /**
   * A thread-safe cache that contains the mapping from each class to all methods in that class and
   * all super-classes, that are annotated with {@code @Subscribe}. The cache is shared across all
   * instances of this class; this greatly improves performance if multiple EventBus instances are
   * created and objects of the same class are registered on all of them.
   */
  private static final LoadingCache<Class<?>, ImmutableList<Method>> subscriberMethodsCache =
      CacheBuilder.newBuilder()
          .weakKeys()
          .build(CacheLoader.from(SubscriberRegistry::getAnnotatedMethodsNotCached));

  /**
   * Returns all subscribers for the given listener grouped by the type of event they subscribe to.
   */
  private Multimap<Class<?>, Subscriber> findAllSubscribers(Object listener) {
    Multimap<Class<?>, Subscriber> methodsInListener = HashMultimap.create();
    Class<?> clazz = listener.getClass();
