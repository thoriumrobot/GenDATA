// Source-based slice around line 112
// Method: <com.google.common.collect.TreeBasedTable: TreeBasedTable create(Comparator,Comparator)>

  }

  /**
   * Creates an empty {@code TreeBasedTable} that is ordered by the specified comparators.
   *
   * @param rowComparator the comparator that orders the row keys
   * @param columnComparator the comparator that orders the column keys
   */
  public static <R, C, V> TreeBasedTable<R, C, V> create(
      Comparator<? super R> rowComparator, Comparator<? super C> columnComparator) {
    checkNotNull(rowComparator);
    checkNotNull(columnComparator);
    return new TreeBasedTable<>(rowComparator, columnComparator);
  }

  /**
   * Creates a {@code TreeBasedTable} with the same mappings and sort order as the specified {@code
   * TreeBasedTable}.
   */
  public static <R, C, V> TreeBasedTable<R, C, V> create(TreeBasedTable<R, C, ? extends V> table) {
