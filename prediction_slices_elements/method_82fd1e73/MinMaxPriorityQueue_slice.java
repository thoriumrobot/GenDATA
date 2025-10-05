// Source-based slice around line 368
// Method: <com.google.common.collect.MinMaxPriorityQueue: E pollLast()>

  public @Nullable E peekFirst() {
    return peek();
  }

  /**
   * Removes and returns the greatest element of this queue, or returns {@code null} if the queue is
   * empty.
   */
  @CanIgnoreReturnValue
  public @Nullable E pollLast() {
    return isEmpty() ? null : removeAndGet(getMaxElementIndex());
  }

  /**
   * Removes and returns the greatest element of this queue.
   *
   * @throws NoSuchElementException if the queue is empty
   */
  @CanIgnoreReturnValue
  public E removeLast() {
