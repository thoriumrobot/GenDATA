// Source-based slice around line 378
// Method: <com.google.common.collect.MinMaxPriorityQueue: E removeLast()>

    return isEmpty() ? null : removeAndGet(getMaxElementIndex());
  }

  /**
   * Removes and returns the greatest element of this queue.
   *
   * @throws NoSuchElementException if the queue is empty
   */
  @CanIgnoreReturnValue
  public E removeLast() {
    if (isEmpty()) {
      throw new NoSuchElementException();
    }
    return removeAndGet(getMaxElementIndex());
  }

  /**
   * Retrieves, but does not remove, the greatest element of this queue, or returns {@code null} if
   * the queue is empty.
   */
