// Source-based slice around line 925
// Method: <com.google.common.collect.MinMaxPriorityQueue: Comparator comparator()>

    arraycopy(queue, 0, copyTo, 0, size);
    return copyTo;
  }

  /**
   * Returns the comparator used to order the elements in this queue. Obeys the general contract of
   * {@link PriorityQueue#comparator}, but returns {@link Ordering#natural} instead of {@code null}
   * to indicate natural ordering.
   */
  public Comparator<? super E> comparator() {
    return minHeap.ordering;
  }

  @VisibleForTesting
  int capacity() {
    return queue.length;
  }

  // Size/capacity-related methods

