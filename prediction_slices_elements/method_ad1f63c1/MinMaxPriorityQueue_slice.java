// Source-based slice around line 152
// Method: <com.google.common.collect.MinMaxPriorityQueue: Builder maximumSize(int)>

  }

  /**
   * Creates and returns a new builder, configured to build {@code MinMaxPriorityQueue} instances
   * that are limited to {@code maximumSize} elements. Each time a queue grows beyond this bound, it
   * immediately removes its greatest element (according to its comparator), which might be the
   * element that was just added.
   */
  @SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
  public static Builder<Comparable> maximumSize(int maximumSize) {
    return new Builder<Comparable>(Ordering.natural()).maximumSize(maximumSize);
  }

  /**
   * The builder class used in creation of min-max priority queues. Instead of constructing one
   * directly, use {@link MinMaxPriorityQueue#orderedBy(Comparator)}, {@link
   * MinMaxPriorityQueue#expectedSize(int)} or {@link MinMaxPriorityQueue#maximumSize(int)}.
   *
   * @param <B> the upper bound on the eventual type that can be produced by this builder (for
   *     example, a {@code Builder<Number>} can produce a {@code Queue<Number>} or {@code
