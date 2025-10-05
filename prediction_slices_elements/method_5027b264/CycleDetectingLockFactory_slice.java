// Source-based slice around line 273
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: WithExplicitOrdering newInstanceWithExplicitOrdering(Class,Policy)>

        : new CycleDetectingReentrantReadWriteLock(new LockGraphNode(lockName), fair);
  }

  // A static mapping from an Enum type to its set of LockGraphNodes.
  private static final ConcurrentMap<
          Class<? extends Enum<?>>, Map<? extends Enum<?>, LockGraphNode>>
      lockGraphNodesPerType = new MapMaker().weakKeys().makeMap();

  /** Creates a {@code CycleDetectingLockFactory.WithExplicitOrdering<E>}. */
  public static <E extends Enum<E>> WithExplicitOrdering<E> newInstanceWithExplicitOrdering(
      Class<E> enumClass, Policy policy) {
    // createNodes maps each enumClass to a Map with the corresponding enum key
    // type.
    checkNotNull(enumClass);
    checkNotNull(policy);
    @SuppressWarnings("unchecked")
    Map<E, LockGraphNode> lockGraphNodes = (Map<E, LockGraphNode>) getOrCreateNodes(enumClass);
    return new WithExplicitOrdering<>(policy, lockGraphNodes);
  }

