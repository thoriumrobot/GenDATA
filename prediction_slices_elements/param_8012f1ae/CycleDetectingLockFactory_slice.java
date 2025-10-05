// Source-based slice around line 286
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: Map getOrCreateNodes(Class)>

    checkNotNull(enumClass);
    checkNotNull(policy);
    @SuppressWarnings("unchecked")
    Map<E, LockGraphNode> lockGraphNodes = (Map<E, LockGraphNode>) getOrCreateNodes(enumClass);
    return new WithExplicitOrdering<>(policy, lockGraphNodes);
  }

  @SuppressWarnings("unchecked")
  private static <E extends Enum<E>> Map<? extends E, LockGraphNode> getOrCreateNodes(
      Class<E> clazz) {
    Map<E, LockGraphNode> existing = (Map<E, LockGraphNode>) lockGraphNodesPerType.get(clazz);
    if (existing != null) {
      return existing;
    }
    Map<E, LockGraphNode> created = createNodes(clazz);
    existing = (Map<E, LockGraphNode>) lockGraphNodesPerType.putIfAbsent(clazz, created);
    return MoreObjects.firstNonNull(existing, created);
  }

  /**
