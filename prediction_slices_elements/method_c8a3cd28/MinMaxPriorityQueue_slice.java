// Source-based slice around line 304
// Method: <com.google.common.collect.MinMaxPriorityQueue: E poll()>


    // Adds the element to the end of the heap and bubbles it up to the correct
    // position.
    heapForIndex(insertIndex).bubbleUp(insertIndex, element);
    return size <= maximumSize || pollLast() != element;
  }

  @CanIgnoreReturnValue
  @Override
  public @Nullable E poll() {
    return isEmpty() ? null : removeAndGet(0);
  }

  @SuppressWarnings("unchecked") // we must carefully only allow Es to get in
  E elementData(int index) {
    /*
     * requireNonNull is safe as long as we're careful to call this method only with populated
     * indexes.
     */
    return (E) requireNonNull(queue[index]);
