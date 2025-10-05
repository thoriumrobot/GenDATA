// Source-based slice around line 238
// Method: com.google.common.collect.MinMaxPriorityQueue.modCount

      return Ordering.from((Comparator<T>) comparator);
    }
  }

  private final Heap minHeap;
  private final Heap maxHeap;
  @VisibleForTesting final int maximumSize;
  private @Nullable Object[] queue;
  private int size;
  private int modCount;

  private MinMaxPriorityQueue(Builder<? super E> builder, int queueSize) {
    Ordering<E> ordering = builder.ordering();
    this.minHeap = new Heap(ordering);
    this.maxHeap = new Heap(ordering.reverse());
    minHeap.otherHeap = maxHeap;
    maxHeap.otherHeap = minHeap;

    this.maximumSize = builder.maximumSize;
    // TODO(kevinb): pad?
