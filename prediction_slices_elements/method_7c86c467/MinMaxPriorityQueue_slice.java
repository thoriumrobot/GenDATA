// Source-based slice around line 351
// Method: <com.google.common.collect.MinMaxPriorityQueue: E removeFirst()>

    return poll();
  }

  /**
   * Removes and returns the least element of this queue.
   *
   * @throws NoSuchElementException if the queue is empty
   */
  @CanIgnoreReturnValue
  public E removeFirst() {
    return remove();
  }

  /**
   * Retrieves, but does not remove, the least element of this queue, or returns {@code null} if the
   * queue is empty.
   */
  public @Nullable E peekFirst() {
    return peek();
  }
