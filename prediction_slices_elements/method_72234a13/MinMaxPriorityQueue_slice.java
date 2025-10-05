// Source-based slice around line 930
// Method: <com.google.common.collect.MinMaxPriorityQueue: int capacity()>

   * Returns the comparator used to order the elements in this queue. Obeys the general contract of
   * {@link PriorityQueue#comparator}, but returns {@link Ordering#natural} instead of {@code null}
   * to indicate natural ordering.
   */
  public Comparator<? super E> comparator() {
    return minHeap.ordering;
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
