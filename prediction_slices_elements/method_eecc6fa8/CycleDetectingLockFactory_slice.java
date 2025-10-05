// Source-based slice around line 729
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: void lockStateChanged(CycleDetectingLock)>

      acquiredLockList.add(node);
    }
  }

  /**
   * CycleDetectingLock implementations must call this method in a {@code finally} clause after any
   * attempt to change the lock state, including both lock and unlock attempts. Failure to do so can
   * result in corrupting the acquireLocks set.
   */
  private static void lockStateChanged(CycleDetectingLock lock) {
    if (!lock.isAcquiredByCurrentThread()) {
      // requireNonNull accommodates Android's @RecentlyNullable annotation on ThreadLocal.get
      List<LockGraphNode> acquiredLockList = requireNonNull(acquiredLocks.get());
      LockGraphNode node = lock.getLockGraphNode();
      // Iterate in reverse because locks are usually locked/unlocked in a
      // LIFO order.
      for (int i = acquiredLockList.size() - 1; i >= 0; i--) {
        if (acquiredLockList.get(i) == node) {
          acquiredLockList.remove(i);
          break;
