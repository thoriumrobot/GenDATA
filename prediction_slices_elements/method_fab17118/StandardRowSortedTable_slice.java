// Source-based slice around line 91
// Method: <com.google.common.collect.StandardRowSortedTable: SortedMap createRowMap()>

   * <p>This method returns a {@link SortedMap}, instead of the {@code Map} specified in the {@link
   * Table} interface.
   */
  @Override
  public SortedMap<R, Map<C, V>> rowMap() {
    return (SortedMap<R, Map<C, V>>) super.rowMap();
  }

  @Override
  SortedMap<R, Map<C, V>> createRowMap() {
    return new RowSortedMap();
  }

  @WeakOuter
  private final class RowSortedMap extends RowMap implements SortedMap<R, Map<C, V>> {
    @Override
    public SortedSet<R> keySet() {
      return (SortedSet<R>) super.keySet();
    }

