// Source-based slice around line 108
// Method: <com.google.common.collect.SparseImmutableTable: int size()>

  }

  @Override
  public ImmutableMap<R, Map<C, V>> rowMap() {
    // Casts without copying.
    return ImmutableMap.copyOf(rowMap);
  }

  @Override
  public int size() {
    return cellRowIndices.length;
  }

  @Override
  Cell<R, C, V> getCell(int index) {
    int rowIndex = cellRowIndices[index];
    Entry<R, ImmutableMap<C, V>> rowEntry = rowMap.entrySet().asList().get(rowIndex);
    ImmutableMap<C, V> row = rowEntry.getValue();
    int columnIndex = cellColumnInRowIndices[index];
    Entry<C, V> colEntry = row.entrySet().asList().get(columnIndex);
