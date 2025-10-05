// Source-based slice around line 156
// Method: <com.google.common.eventbus.SubscriberRegistry: Multimap findAllSubscribers(Object)>

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
    for (Method method : getAnnotatedMethods(clazz)) {
      Class<?>[] parameterTypes = method.getParameterTypes();
      Class<?> eventType = parameterTypes[0];
      methodsInListener.put(eventType, Subscriber.create(bus, listener, method));
    }
    return methodsInListener;
  }

