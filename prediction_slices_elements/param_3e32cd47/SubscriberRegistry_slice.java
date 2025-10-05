// Source-based slice around line 193
// Method: <com.google.common.eventbus.SubscriberRegistry: ImmutableList getAnnotatedMethodsNotCached(Class)>

       * UncheckedExecutionException, which has the stack trace from this thread and which has its
       * cause set to the underlying exception (which may be from another thread). If we someday
       * learn that some other exception besides IllegalArgumentException is common, then we could
       * add another special case to throw an instance of it, too.
       */
      throw e;
    }
  }

  private static ImmutableList<Method> getAnnotatedMethodsNotCached(Class<?> clazz) {
    Set<? extends Class<?>> supertypes = TypeToken.of(clazz).getTypes().rawTypes();
    Map<MethodIdentifier, Method> identifiers = new HashMap<>();
    for (Class<?> supertype : supertypes) {
      for (Method method : supertype.getDeclaredMethods()) {
        if (method.isAnnotationPresent(Subscribe.class) && !method.isSynthetic()) {
          // TODO(cgdecker): Should check for a generic parameter type and error out
          Class<?>[] parameterTypes = method.getParameterTypes();
          checkArgument(
              parameterTypes.length == 1,
              "Method %s has @Subscribe annotation but has %s parameters. "
