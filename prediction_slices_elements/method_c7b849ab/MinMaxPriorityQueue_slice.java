// Source-based slice around line 341
// Method: <com.google.common.collect.MinMaxPriorityQueue: E pollFirst()>

        return (maxHeap.compareElements(1, 2) <= 0) ? 1 : 2;
    }
  }

  /**
   * Removes and returns the least element of this queue, or returns {@code null} if the queue is
   * empty.
   */
  @CanIgnoreReturnValue
  public @Nullable E pollFirst() {
    return poll();
  }

  /**
   * Removes and returns the least element of this queue.
   *
   * @throws NoSuchElementException if the queue is empty
   */
  @CanIgnoreReturnValue
  public E removeFirst() {
