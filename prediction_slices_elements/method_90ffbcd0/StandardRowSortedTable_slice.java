// Source-based slice around line 64
// Method: <com.google.common.collect.StandardRowSortedTable: SortedMap sortedBackingMap()>

   * which return a Table view with rows keys in a given range. Create a
   * RowSortedTable subinterface with the revised methods?
   */

  StandardRowSortedTable(
      SortedMap<R, Map<C, V>> backingMap, Supplier<? extends Map<C, V>> factory) {
    super(backingMap, factory);
  }

  private SortedMap<R, Map<C, V>> sortedBackingMap() {
    return (SortedMap<R, Map<C, V>>) backingMap;
  }

  /**
   * {@inheritDoc}
   *
   * <p>This method returns a {@link SortedSet}, instead of the {@code Set} specified in the {@link
   * Table} interface.
   */
  @Override
