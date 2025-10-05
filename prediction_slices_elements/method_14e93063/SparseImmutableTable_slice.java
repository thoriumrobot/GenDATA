// Source-based slice around line 133
// Method: <com.google.common.collect.SparseImmutableTable: Object writeReplace()>

    int rowIndex = cellRowIndices[index];
    ImmutableMap<C, V> row = rowMap.values().asList().get(rowIndex);
    int columnIndex = cellColumnInRowIndices[index];
    return row.values().asList().get(columnIndex);
  }

  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
    Map<C, Integer> columnKeyToIndex = Maps.indexMap(columnKeySet());
    int[] cellColumnIndices = new int[cellSet().size()];
    int i = 0;
    for (Cell<R, C, V> cell : cellSet()) {
      // requireNonNull is safe because the cell exists in the table.
      cellColumnIndices[i++] = requireNonNull(columnKeyToIndex.get(cell.getColumnKey()));
    }
    return SerializedForm.create(this, cellRowIndices, cellColumnIndices);
  }
}
