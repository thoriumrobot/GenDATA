// Source-based slice around line 93
// Method: <com.google.common.collect.EvictingQueue: boolean offer(E)>


  /**
   * Adds the given element to this queue. If the queue is currently full, the element at the head
   * of the queue is evicted to make room.
   *
   * @return {@code true} always
   */
  @Override
  @CanIgnoreReturnValue
  public boolean offer(E e) {
    return add(e);
  }

  /**
   * Adds the given element to this queue. If the queue is currently full, the element at the head
   * of the queue is evicted to make room.
   *
   * @return {@code true} always
   */
  @Override
