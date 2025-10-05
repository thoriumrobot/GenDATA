// Source-based slice around line 463
// Method: com.google.common.util.concurrent.CycleDetectingLockFactory.acquiredLocks

    this.policy = checkNotNull(policy);
  }

  /**
   * Tracks the currently acquired locks for each Thread, kept up to date by calls to {@link
   * #aboutToAcquire(CycleDetectingLock)} and {@link #lockStateChanged(CycleDetectingLock)}.
   */
  // This is logically a Set, but an ArrayList is used to minimize the amount
  // of allocation done on lock()/unlock().
  private static final ThreadLocal<List<LockGraphNode>> acquiredLocks =
      new ThreadLocal<List<LockGraphNode>>() {
        @Override
        protected List<LockGraphNode> initialValue() {
          return newArrayListWithCapacity(3);
        }
      };

  /**
   * A Throwable used to record a stack trace that illustrates an example of a specific lock
   * acquisition ordering. The top of the stack trace is truncated such that it starts with the
