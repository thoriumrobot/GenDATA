// Source-based slice around line 451
// Method: com.google.common.util.concurrent.CycleDetectingLockFactory.policy

          : new CycleDetectingReentrantReadWriteLock(
              requireNonNull(lockGraphNodes.get(rank)), fair);
    }
  }

  //////// Implementation /////////

  private static final LazyLogger logger = new LazyLogger(CycleDetectingLockFactory.class);

  final Policy policy;

  private CycleDetectingLockFactory(Policy policy) {
    this.policy = checkNotNull(policy);
  }

  /**
   * Tracks the currently acquired locks for each Thread, kept up to date by calls to {@link
   * #aboutToAcquire(CycleDetectingLock)} and {@link #lockStateChanged(CycleDetectingLock)}.
   */
  // This is logically a Set, but an ArrayList is used to minimize the amount
