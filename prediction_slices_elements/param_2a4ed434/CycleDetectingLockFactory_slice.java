// Source-based slice around line 237
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: ReentrantLock newReentrantLock(String)>

    };
  }

  /** Creates a new factory with the specified policy. */
  public static CycleDetectingLockFactory newInstance(Policy policy) {
    return new CycleDetectingLockFactory(policy);
  }

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
