// Source-based slice around line 449
// Method: com.google.common.util.concurrent.CycleDetectingLockFactory.logger

          // requireNonNull is safe because createNodes inserts an entry for every E.
          // (If the caller passes `null` for the `rank` parameter, this will throw, but that's OK.)
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
