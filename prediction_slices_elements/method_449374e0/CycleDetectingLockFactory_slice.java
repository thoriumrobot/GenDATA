// Source-based slice around line 303
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: Map createNodes(Class)>

  }

  /**
   * For a given Enum type, creates an immutable map from each of the Enum's values to a
   * corresponding LockGraphNode, with the {@code allowedPriorLocks} and {@code
   * disallowedPriorLocks} prepopulated with nodes according to the natural ordering of the
   * associated Enum values.
   */
  @VisibleForTesting
  static <E extends Enum<E>> Map<E, LockGraphNode> createNodes(Class<E> clazz) {
    EnumMap<E, LockGraphNode> map = Maps.newEnumMap(clazz);
    E[] keys = clazz.getEnumConstants();
    int numKeys = keys.length;
    ArrayList<LockGraphNode> nodes = Lists.newArrayListWithCapacity(numKeys);
    // Create a LockGraphNode for each enum value.
    for (E key : keys) {
      LockGraphNode node = new LockGraphNode(getLockName(key));
      nodes.add(node);
      map.put(key, node);
    }
