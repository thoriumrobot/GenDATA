// Source-based slice around line 74
// Method: <com.google.common.eventbus.SubscriberRegistry: void register(Object)>


  /** The event bus this registry belongs to. */
  @Weak private final EventBus bus;

  SubscriberRegistry(EventBus bus) {
    this.bus = checkNotNull(bus);
  }

  /** Registers all subscriber methods on the given listener object. */
  void register(Object listener) {
    Multimap<Class<?>, Subscriber> listenerMethods = findAllSubscribers(listener);

    for (Entry<Class<?>, Collection<Subscriber>> entry : listenerMethods.asMap().entrySet()) {
      Class<?> eventType = entry.getKey();
      Collection<Subscriber> eventMethodsInListener = entry.getValue();

      CopyOnWriteArraySet<Subscriber> eventSubscribers = subscribers.get(eventType);

      if (eventSubscribers == null) {
        CopyOnWriteArraySet<Subscriber> newSet = new CopyOnWriteArraySet<>();
