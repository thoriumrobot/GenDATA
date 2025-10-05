// Source-based slice around line 165
// Method: <com.google.common.collect.TreeBasedTable: Comparator columnComparator()>

  /**
   * Returns the comparator that orders the columns. With natural ordering, {@link
   * Ordering#natural()} is returned.
   *
   * @deprecated Store the {@link Comparator} alongside the {@link Table}. Or, if you know that the
   *     {@link Table} contains at least one value, you can retrieve the {@link Comparator} with:
   *     {@code ((SortedMap<C, V>) table.rowMap().values().iterator().next()).comparator();}.
   */
  @Deprecated
  public Comparator<? super C> columnComparator() {
    return columnComparator;
  }

  // TODO(lowasser): make column return a SortedMap

  /**
   * {@inheritDoc}
   *
   * <p>Because a {@code TreeBasedTable} has unique sorted values for a given row, this method
   * returns a {@link SortedMap}, instead of the {@link Map} specified in the {@link Table}
