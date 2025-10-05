// Source-based slice around line 233
// Method: com.google.common.collect.MinMaxPriorityQueue.minHeap

      return queue;
    }

    @SuppressWarnings("unchecked") // safe "contravariant cast"
    private <T extends B> Ordering<T> ordering() {
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
