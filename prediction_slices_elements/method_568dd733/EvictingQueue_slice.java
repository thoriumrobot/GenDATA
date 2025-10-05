// Source-based slice around line 66
// Method: <com.google.common.collect.EvictingQueue: EvictingQueue create(int)>

    this.maxSize = maxSize;
  }

  /**
   * Creates and returns a new evicting queue that will hold up to {@code maxSize} elements.
   *
   * <p>When {@code maxSize} is zero, elements will be evicted immediately after being added to the
   * queue.
   */
  public static <E> EvictingQueue<E> create(int maxSize) {
    return new EvictingQueue<>(maxSize);
  }

  /**
   * Returns the number of additional elements that this queue can accept without evicting; zero if
   * the queue is currently full.
   *
   * @since 16.0
   */
  public int remainingCapacity() {
