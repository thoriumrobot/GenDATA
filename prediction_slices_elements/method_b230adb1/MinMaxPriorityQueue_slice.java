// Source-based slice around line 117
// Method: <com.google.common.collect.MinMaxPriorityQueue: MinMaxPriorityQueue create(Iterable)>

   */
  public static <E extends Comparable<E>> MinMaxPriorityQueue<E> create() {
    return new Builder<Comparable<E>>(Ordering.natural()).create();
  }

  /**
   * Creates a new min-max priority queue using natural order, no maximum size, and initially
   * containing the given elements.
   */
  public static <E extends Comparable<E>> MinMaxPriorityQueue<E> create(
      Iterable<? extends E> initialContents) {
    return new Builder<E>(Ordering.natural()).create(initialContents);
  }

  /**
   * Creates and returns a new builder, configured to build {@code MinMaxPriorityQueue} instances
   * that use {@code comparator} to determine the least and greatest elements.
   */
  /*
   * TODO(cpovirk): Change to Comparator<? super B> to permit Comparator<@Nullable ...> and
