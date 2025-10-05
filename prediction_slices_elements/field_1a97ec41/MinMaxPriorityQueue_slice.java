// Source-based slice around line 488
// Method: com.google.common.collect.MinMaxPriorityQueue.ODD_POWERS_OF_TWO

    removeAt(index);
    return value;
  }

  private Heap heapForIndex(int i) {
    return isEvenLevel(i) ? minHeap : maxHeap;
  }

  private static final int EVEN_POWERS_OF_TWO = 0x55555555;
  private static final int ODD_POWERS_OF_TWO = 0xaaaaaaaa;

  @VisibleForTesting
  static boolean isEvenLevel(int index) {
    int oneBased = ~~(index + 1); // for GWT
    checkState(oneBased > 0, "negative index");
    return (oneBased & EVEN_POWERS_OF_TWO) > (oneBased & ODD_POWERS_OF_TWO);
  }

  /**
   * Returns {@code true} if the MinMax heap structure holds. This is only used in testing.
