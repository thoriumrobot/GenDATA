// Source-based slice around line 681
// Method: <com.google.common.collect.StandardTable: Set columnKeySet()>

  /**
   * {@inheritDoc}
   *
   * <p>The returned set has an iterator that does not support {@code remove()}.
   *
   * <p>The set's iterator traverses the columns of the first row, the columns of the second row,
   * etc., skipping any columns that have appeared previously.
   */
  @Override
  public Set<C> columnKeySet() {
    Set<C> result = columnKeySet;
    return (result == null) ? columnKeySet = new ColumnKeySet() : result;
  }

  @WeakOuter
  private final class ColumnKeySet extends TableSet<C> {
    @Override
    public Iterator<C> iterator() {
      return createColumnKeyIterator();
    }
