// Source-based slice around line 132
// Method: <com.google.common.collect.MinMaxPriorityQueue: Builder orderedBy(Comparator)>

   * Creates and returns a new builder, configured to build {@code MinMaxPriorityQueue} instances
   * that use {@code comparator} to determine the least and greatest elements.
   */
  /*
   * TODO(cpovirk): Change to Comparator<? super B> to permit Comparator<@Nullable ...> and
   * Comparator<SupertypeOfB>? What we have here matches the immutable collections, but those also
   * expose a public Builder constructor that accepts "? super." So maybe we should do *that*
   * instead.
   */
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
