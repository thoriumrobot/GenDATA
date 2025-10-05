// Source-based slice around line 241
// Method: <com.google.common.eventbus.SubscriberRegistry: ImmutableSet flattenHierarchy(Class)>

              CacheLoader.from(
                  concreteClass ->
                      ImmutableSet.copyOf(TypeToken.of(concreteClass).getTypes().rawTypes())));

  /**
   * Flattens a class's type hierarchy into a set of {@code Class} objects including all
   * superclasses (transitively) and all interfaces implemented by these superclasses.
   */
  @VisibleForTesting
  static ImmutableSet<Class<?>> flattenHierarchy(Class<?> concreteClass) {
    return flattenHierarchyCache.getUnchecked(concreteClass);
  }

  private static final class MethodIdentifier {

    private final String name;
    private final List<Class<?>> parameterTypes;

    MethodIdentifier(Method method) {
      this.name = method.getName();
