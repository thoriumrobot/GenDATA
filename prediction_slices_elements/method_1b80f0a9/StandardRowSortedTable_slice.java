// Source-based slice around line 75
// Method: <com.google.common.collect.StandardRowSortedTable: SortedSet rowKeySet()>

  }

  /**
   * {@inheritDoc}
   *
   * <p>This method returns a {@link SortedSet}, instead of the {@code Set} specified in the {@link
   * Table} interface.
   */
  @Override
  public SortedSet<R> rowKeySet() {
    return (SortedSet<R>) rowMap().keySet();
  }

  /**
   * {@inheritDoc}
   *
   * <p>This method returns a {@link SortedMap}, instead of the {@code Map} specified in the {@link
   * Table} interface.
   */
  @Override
