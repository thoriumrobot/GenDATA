// Source-based slice around line 252
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: ReentrantReadWriteLock newReentrantReadWriteLock(String)>

   * the warning or exception output to help identify the locks involved in the detected deadlock.
   */
  public ReentrantLock newReentrantLock(String lockName, boolean fair) {
    return policy == Policies.DISABLED
        ? new ReentrantLock(fair)
        : new CycleDetectingReentrantLock(new LockGraphNode(lockName), fair);
  }

  /** Equivalent to {@code newReentrantReadWriteLock(lockName, false)}. */
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
