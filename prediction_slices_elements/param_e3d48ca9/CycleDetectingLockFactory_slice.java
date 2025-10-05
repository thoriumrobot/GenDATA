// Source-based slice around line 261
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: ReentrantReadWriteLock newReentrantReadWriteLock(String,boolean)>

  public ReentrantReadWriteLock newReentrantReadWriteLock(String lockName) {
    return newReentrantReadWriteLock(lockName, false);
  }

  /**
   * Creates a {@link ReentrantReadWriteLock} with the given fairness policy. The {@code lockName}
   * is used in the warning or exception output to help identify the locks involved in the detected
   * deadlock.
   */
  public ReentrantReadWriteLock newReentrantReadWriteLock(String lockName, boolean fair) {
    return policy == Policies.DISABLED
        ? new ReentrantReadWriteLock(fair)
        : new CycleDetectingReentrantReadWriteLock(new LockGraphNode(lockName), fair);
  }

  // A static mapping from an Enum type to its set of LockGraphNodes.
  private static final ConcurrentMap<
          Class<? extends Enum<?>>, Map<? extends Enum<?>, LockGraphNode>>
      lockGraphNodesPerType = new MapMaker().weakKeys().makeMap();

