// Source-based slice around line 253
// Method: <com.google.common.collect.MinMaxPriorityQueue: int size()>

    minHeap.otherHeap = maxHeap;
    maxHeap.otherHeap = minHeap;

    this.maximumSize = builder.maximumSize;
    // TODO(kevinb): pad?
    this.queue = new Object[queueSize];
  }

  @Override
  public int size() {
    return size;
  }

  /**
   * Adds the given element to this queue. If this queue has a maximum size, after adding {@code
   * element} the queue will automatically evict its greatest element (according to its comparator),
   * which may be {@code element} itself.
   *
   * @return {@code true} always
   */
