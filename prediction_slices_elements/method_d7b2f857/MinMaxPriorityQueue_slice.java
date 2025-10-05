// Source-based slice around line 503
// Method: <com.google.common.collect.MinMaxPriorityQueue: boolean isIntact()>

    return (oneBased & EVEN_POWERS_OF_TWO) > (oneBased & ODD_POWERS_OF_TWO);
  }

  /**
   * Returns {@code true} if the MinMax heap structure holds. This is only used in testing.
   *
   * <p>TODO(kevinb): move to the test class?
   */
  @VisibleForTesting
  boolean isIntact() {
    for (int i = 1; i < size; i++) {
      if (!heapForIndex(i).verifyIndex(i)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Each instance of MinMaxPriorityQueue encapsulates two instances of Heap: a min-heap and a
