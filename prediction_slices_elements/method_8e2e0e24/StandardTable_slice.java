// Source-based slice around line 236
// Method: <com.google.common.collect.StandardTable: Set cellSet()>

   * {@inheritDoc}
   *
   * <p>The set's iterator traverses the mappings for the first row, the mappings for the second
   * row, and so on.
   *
   * <p>Each cell is an immutable snapshot of a row key / column key / value mapping, taken at the
   * time the cell is returned by a method call to the set or its iterator.
   */
  @Override
  public Set<Cell<R, C, V>> cellSet() {
    return super.cellSet();
  }

  @Override
  Iterator<Cell<R, C, V>> cellIterator() {
    return new CellIterator();
  }

  private final class CellIterator implements Iterator<Cell<R, C, V>> {
    final Iterator<Entry<R, Map<C, V>>> rowIterator = backingMap.entrySet().iterator();
