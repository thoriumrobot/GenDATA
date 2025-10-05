// Source-based slice around line 123
// Method: <com.google.common.collect.SparseImmutableTable: V getValue(int)>

    int rowIndex = cellRowIndices[index];
    Entry<R, ImmutableMap<C, V>> rowEntry = rowMap.entrySet().asList().get(rowIndex);
    ImmutableMap<C, V> row = rowEntry.getValue();
    int columnIndex = cellColumnInRowIndices[index];
    Entry<C, V> colEntry = row.entrySet().asList().get(columnIndex);
    return cellOf(rowEntry.getKey(), colEntry.getKey(), colEntry.getValue());
  }

  @Override
  V getValue(int index) {
    int rowIndex = cellRowIndices[index];
    ImmutableMap<C, V> row = rowMap.values().asList().get(rowIndex);
    int columnIndex = cellColumnInRowIndices[index];
    return row.values().asList().get(columnIndex);
  }

  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
