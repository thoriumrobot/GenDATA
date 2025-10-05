// Source-based slice around line 76
// Method: <com.google.common.collect.EvictingQueue: int remainingCapacity()>

    return new EvictingQueue<>(maxSize);
  }

  /**
   * Returns the number of additional elements that this queue can accept without evicting; zero if
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
