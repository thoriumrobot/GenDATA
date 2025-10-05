// Source-based slice around line 487
// Method: com.google.common.collect.MinMaxPriorityQueue.EVEN_POWERS_OF_TWO

    E value = elementData(index);
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
