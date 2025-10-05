// Source-based slice around line 245
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: ReentrantLock newReentrantLock(String,boolean)>

  /** Equivalent to {@code newReentrantLock(lockName, false)}. */
  public ReentrantLock newReentrantLock(String lockName) {
    return newReentrantLock(lockName, false);
  }

  /**
   * Creates a {@link ReentrantLock} with the given fairness policy. The {@code lockName} is used in
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

