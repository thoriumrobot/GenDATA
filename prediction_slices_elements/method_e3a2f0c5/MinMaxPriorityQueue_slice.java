// Source-based slice around line 318
// Method: <com.google.common.collect.MinMaxPriorityQueue: E peek()>

  E elementData(int index) {
    /*
     * requireNonNull is safe as long as we're careful to call this method only with populated
     * indexes.
     */
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
