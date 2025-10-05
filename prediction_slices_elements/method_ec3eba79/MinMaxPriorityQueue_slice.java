// Source-based slice around line 323
// Method: <com.google.common.collect.MinMaxPriorityQueue: int getMaxElementIndex()>

    return (E) requireNonNull(queue[index]);
  }

  @Override
  public @Nullable E peek() {
    return isEmpty() ? null : elementData(0);
  }

  /** Returns the index of the max element. */
  private int getMaxElementIndex() {
    switch (size) {
      case 1:
        return 0; // The lone element in the queue is the maximum.
      case 2:
        return 1; // The lone element in the maxHeap is the maximum.
      default:
        // The max element must sit on the first level of the maxHeap. It is
        // actually the *lesser* of the two from the maxHeap's perspective.
        return (maxHeap.compareElements(1, 2) <= 0) ? 1 : 2;
    }
