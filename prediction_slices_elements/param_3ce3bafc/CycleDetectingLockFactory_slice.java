// Source-based slice around line 714
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: void aboutToAcquire(CycleDetectingLock)>

        }
      }
      return null;
    }
  }

  /**
   * CycleDetectingLock implementations must call this method before attempting to acquire the lock.
   */
  private void aboutToAcquire(CycleDetectingLock lock) {
    if (!lock.isAcquiredByCurrentThread()) {
      // requireNonNull accommodates Android's @RecentlyNullable annotation on ThreadLocal.get
      List<LockGraphNode> acquiredLockList = requireNonNull(acquiredLocks.get());
      LockGraphNode node = lock.getLockGraphNode();
      node.checkAcquiredLocks(policy, acquiredLockList);
      acquiredLockList.add(node);
    }
  }

  /**
