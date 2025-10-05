// Source-based slice around line 477
// Method: <com.google.common.collect.MinMaxPriorityQueue: E removeAndGet(int)>

    final E replaced;

    MoveDesc(E toTrickle, E replaced) {
      this.toTrickle = toTrickle;
      this.replaced = replaced;
    }
  }

  /** Removes and returns the value at {@code index}. */
  private E removeAndGet(int index) {
    E value = elementData(index);
    removeAt(index);
    return value;
  }

  private Heap heapForIndex(int i) {
    return isEvenLevel(i) ? minHeap : maxHeap;
  }

  private static final int EVEN_POWERS_OF_TWO = 0x55555555;
