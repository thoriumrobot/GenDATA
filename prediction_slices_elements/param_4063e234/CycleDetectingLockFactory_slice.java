// Source-based slice around line 232
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: CycleDetectingLockFactory newInstance(Policy)>

     * cycle detection performed by locks created by other factories.
     */
    DISABLED {
      @Override
      public void handlePotentialDeadlock(PotentialDeadlockException e) {}
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
