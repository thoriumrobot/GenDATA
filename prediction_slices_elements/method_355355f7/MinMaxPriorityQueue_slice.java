// Source-based slice around line 359
// Method: <com.google.common.collect.MinMaxPriorityQueue: E peekFirst()>

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

  /**
   * Removes and returns the greatest element of this queue, or returns {@code null} if the queue is
   * empty.
   */
  @CanIgnoreReturnValue
  public @Nullable E pollLast() {
    return isEmpty() ? null : removeAndGet(getMaxElementIndex());
