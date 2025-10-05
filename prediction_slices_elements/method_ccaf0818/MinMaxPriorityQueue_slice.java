// Source-based slice around line 141
// Method: <com.google.common.collect.MinMaxPriorityQueue: Builder expectedSize(int)>

  public static <B> Builder<B> orderedBy(Comparator<B> comparator) {
    return new Builder<>(comparator);
  }

  /**
   * Creates and returns a new builder, configured to build {@code MinMaxPriorityQueue} instances
   * sized appropriately to hold {@code expectedSize} elements.
   */
  @SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
  public static Builder<Comparable> expectedSize(int expectedSize) {
    return new Builder<Comparable>(Ordering.natural()).expectedSize(expectedSize);
  }

  /**
   * Creates and returns a new builder, configured to build {@code MinMaxPriorityQueue} instances
   * that are limited to {@code maximumSize} elements. Each time a queue grows beyond this bound, it
   * immediately removes its greatest element (according to its comparator), which might be the
   * element that was just added.
   */
  @SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
