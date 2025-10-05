// Source-based slice around line 228
// Method: com.google.common.eventbus.SubscriberRegistry.flattenHierarchyCache

            identifiers.put(ident, method);
          }
        }
      }
    }
    return ImmutableList.copyOf(identifiers.values());
  }

  /** Global cache of classes to their flattened hierarchy of supertypes. */
  private static final LoadingCache<Class<?>, ImmutableSet<Class<?>>> flattenHierarchyCache =
      CacheBuilder.newBuilder()
          .weakKeys()
          .build(
              CacheLoader.from(
                  concreteClass ->
                      ImmutableSet.copyOf(TypeToken.of(concreteClass).getTypes().rawTypes())));

  /**
   * Flattens a class's type hierarchy into a set of {@code Class} objects including all
   * superclasses (transitively) and all interfaces implemented by these superclasses.
