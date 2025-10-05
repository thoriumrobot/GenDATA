// Source-based slice around line 101
// Method: <com.google.common.collect.TreeBasedTable: TreeBasedTable create()>

  /**
   * Creates an empty {@code TreeBasedTable} that uses the natural orderings of both row and column
   * keys.
   *
   * <p>The method signature specifies {@code R extends Comparable} with a raw {@link Comparable},
   * instead of {@code R extends Comparable<? super R>}, and the same for {@code C}. That's
   * necessary to support classes defined without generics.
   */
  @SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
  public static <R extends Comparable, C extends Comparable, V> TreeBasedTable<R, C, V> create() {
    return new TreeBasedTable<>(Ordering.natural(), Ordering.natural());
  }

  /**
   * Creates an empty {@code TreeBasedTable} that is ordered by the specified comparators.
   *
   * @param rowComparator the comparator that orders the row keys
   * @param columnComparator the comparator that orders the column keys
   */
  public static <R, C, V> TreeBasedTable<R, C, V> create(
