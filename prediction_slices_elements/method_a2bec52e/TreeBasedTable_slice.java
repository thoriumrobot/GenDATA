// Source-based slice around line 122
// Method: <com.google.common.collect.TreeBasedTable: TreeBasedTable create(TreeBasedTable)>

    checkNotNull(rowComparator);
    checkNotNull(columnComparator);
    return new TreeBasedTable<>(rowComparator, columnComparator);
  }

  /**
   * Creates a {@code TreeBasedTable} with the same mappings and sort order as the specified {@code
   * TreeBasedTable}.
   */
  public static <R, C, V> TreeBasedTable<R, C, V> create(TreeBasedTable<R, C, ? extends V> table) {
    TreeBasedTable<R, C, V> result =
        // requireNonNull is safe, as discussed in rowComparator() below.
        new TreeBasedTable<>(
            requireNonNull(table.rowKeySet().comparator()), table.columnComparator());
    result.putAll(table);
    return result;
  }

  TreeBasedTable(Comparator<? super R> rowComparator, Comparator<? super C> columnComparator) {
    super(new TreeMap<R, Map<C, V>>(rowComparator), new Factory<C, V>(columnComparator));
