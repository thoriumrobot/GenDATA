// Source-based slice around line 289
// Method: <com.google.common.collect.MinMaxPriorityQueue: boolean offer(E)>

  }

  /**
   * Adds the given element to this queue. If this queue has a maximum size, after adding {@code
   * element} the queue will automatically evict its greatest element (according to its comparator),
   * which may be {@code element} itself.
   */
  @CanIgnoreReturnValue
  @Override
  public boolean offer(E element) {
    checkNotNull(element);
    modCount++;
    int insertIndex = size++;

    growIfNeeded();

    // Adds the element to the end of the heap and bubbles it up to the correct
    // position.
    heapForIndex(insertIndex).bubbleUp(insertIndex, element);
    return size <= maximumSize || pollLast() != element;
