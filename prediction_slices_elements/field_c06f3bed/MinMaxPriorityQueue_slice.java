// Source-based slice around line 936
// Method: com.google.common.collect.MinMaxPriorityQueue.DEFAULT_CAPACITY

  }

  @VisibleForTesting
  int capacity() {
    return queue.length;
  }

  // Size/capacity-related methods

  private static final int DEFAULT_CAPACITY = 11;

  @VisibleForTesting
  static int initialQueueSize(
      int configuredExpectedSize, int maximumSize, Iterable<?> initialContents) {
    // Start with what they said, if they said it, otherwise DEFAULT_CAPACITY
    int result =
        (configuredExpectedSize == Builder.UNSET_EXPECTED_SIZE)
            ? DEFAULT_CAPACITY
            : configuredExpectedSize;

