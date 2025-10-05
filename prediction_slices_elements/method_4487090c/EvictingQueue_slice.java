// Source-based slice around line 81
// Method: <com.google.common.collect.EvictingQueue: Queue delegate()>

   * the queue is currently full.
   *
   * @since 16.0
   */
  public int remainingCapacity() {
    return maxSize - size();
  }

  @Override
  protected Queue<E> delegate() {
    return delegate;
  }

  /**
   * Adds the given element to this queue. If the queue is currently full, the element at the head
   * of the queue is evicted to make room.
   *
   * @return {@code true} always
   */
  @Override
