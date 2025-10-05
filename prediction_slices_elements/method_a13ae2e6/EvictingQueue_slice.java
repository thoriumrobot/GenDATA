// Source-based slice around line 105
// Method: <com.google.common.collect.EvictingQueue: boolean add(E)>


  /**
   * Adds the given element to this queue. If the queue is currently full, the element at the head
   * of the queue is evicted to make room.
   *
   * @return {@code true} always
   */
  @Override
  @CanIgnoreReturnValue
  public boolean add(E e) {
    checkNotNull(e); // check before removing
    if (maxSize == 0) {
      return true;
    }
    if (size() == maxSize) {
      delegate.remove();
    }
    delegate.add(e);
    return true;
  }
