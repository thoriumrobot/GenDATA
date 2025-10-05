// Source-based slice around line 167
// Method: <com.google.common.eventbus.SubscriberRegistry: ImmutableList getAnnotatedMethods(Class)>

    Class<?> clazz = listener.getClass();
    for (Method method : getAnnotatedMethods(clazz)) {
      Class<?>[] parameterTypes = method.getParameterTypes();
      Class<?> eventType = parameterTypes[0];
      methodsInListener.put(eventType, Subscriber.create(bus, listener, method));
    }
    return methodsInListener;
  }

  private static ImmutableList<Method> getAnnotatedMethods(Class<?> clazz) {
    try {
      return subscriberMethodsCache.getUnchecked(clazz);
    } catch (UncheckedExecutionException e) {
      if (e.getCause() instanceof IllegalArgumentException) {
        /*
         * IllegalArgumentException is the one unchecked exception that we know is likely to happen
         * (thanks to the checkArgument calls in getAnnotatedMethodsNotCached). If it happens, we'd
         * prefer to propagate an IllegalArgumentException to the caller. However, we don't want to
         * simply rethrow an exception (e.getCause()) that may in rare cases have come from another
         * thread. To accomplish both goals, we wrap that IllegalArgumentException in a new
